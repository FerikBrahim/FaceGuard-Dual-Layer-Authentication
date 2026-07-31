from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from faceguard.io import load_rgb
from faceguard.siamese.model import SiameseVerifier

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class DeepfakeTrial:
    identity: str
    method: str
    path: str
    similarity: float
    distance: float
    is_attack: bool


class DeepfakeEvalDataset:
    """Structured real/fake dataset used by the uploaded deepfake notebook.

    Required layout::

        root/real/<identity>/*
        root/fake/<method>/<identity>/*
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.real_root = self.root / "real"
        self.fake_root = self.root / "fake"
        if not self.real_root.exists() or not self.fake_root.exists():
            raise FileNotFoundError("Expected root/real and root/fake directories")
        self.real_images: dict[str, list[Path]] = {
            directory.name: self._images(directory)
            for directory in sorted(self.real_root.iterdir())
            if directory.is_dir() and self._images(directory)
        }
        self.fake_images: list[tuple[Path, str, str]] = []
        for method_dir in sorted(path for path in self.fake_root.iterdir() if path.is_dir()):
            for identity_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
                for image in self._images(identity_dir):
                    self.fake_images.append((image, identity_dir.name, method_dir.name))

    @staticmethod
    def _images(root: Path) -> list[Path]:
        return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def evaluate_deepfake_robustness(
    verifier: SiameseVerifier,
    dataset: DeepfakeEvalDataset,
    n_reference_frames: int = 5,
    max_real_per_identity: int = 100,
    max_fake_per_identity_method: int = 100,
    seed: int = 2026,
) -> list[DeepfakeTrial]:
    """Compare identity templates with held-out real and claimed-identity fake frames."""
    rng = np.random.default_rng(seed)
    trials: list[DeepfakeTrial] = []
    references: dict[str, np.ndarray] = {}

    # The verifier API compares images directly. Use the first selected reference
    # image for a deterministic single-template evaluation consistent with the CLI.
    reference_paths: dict[str, Path] = {}
    for identity, paths in dataset.real_images.items():
        if len(paths) < n_reference_frames + 1:
            continue
        order = rng.permutation(len(paths))
        selected = [paths[index] for index in order]
        reference_paths[identity] = selected[0]
        reference = load_rgb(selected[0])
        for path in selected[n_reference_frames : n_reference_frames + max_real_per_identity]:
            result = verifier.verify(reference, load_rgb(path))
            trials.append(
                DeepfakeTrial(identity, "genuine", str(path), result.similarity, result.distance, False)
            )

    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path, identity, method in dataset.fake_images:
        if identity in reference_paths:
            grouped[(identity, method)].append(path)
    for (identity, method), paths in sorted(grouped.items()):
        reference = load_rgb(reference_paths[identity])
        order = rng.permutation(len(paths))
        for index in order[:max_fake_per_identity_method]:
            path = paths[int(index)]
            result = verifier.verify(reference, load_rgb(path))
            trials.append(DeepfakeTrial(identity, method, str(path), result.similarity, result.distance, True))
    return trials


def split_trials_by_identity(
    trials: Sequence[DeepfakeTrial], calibration_fraction: float = 0.5, seed: int = 2026
) -> tuple[list[DeepfakeTrial], list[DeepfakeTrial]]:
    identities = sorted({trial.identity for trial in trials})
    if len(identities) < 2:
        raise ValueError("at least two identities are required for an identity-disjoint split")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(identities, dtype=object)[rng.permutation(len(identities))]
    count = max(1, min(len(identities) - 1, int(round(len(identities) * calibration_fraction))))
    calibration_ids = set(shuffled[:count].astype(str))
    calibration = [trial for trial in trials if trial.identity in calibration_ids]
    test = [trial for trial in trials if trial.identity not in calibration_ids]
    return calibration, test


def anti_spoofing_metrics(trials: Sequence[DeepfakeTrial], similarity_threshold: float) -> dict:
    """ISO/IEC 30107-3-style APCER/BPCER plus threshold-free AUC/EER."""
    if not trials:
        raise ValueError("no trials supplied")
    labels = np.asarray([int(trial.is_attack) for trial in trials], dtype=np.int32)
    similarities = np.asarray([trial.similarity for trial in trials], dtype=np.float64)
    accepted = similarities >= float(similarity_threshold)
    attack = labels == 1
    bona_fide = labels == 0
    apcer = float(np.mean(accepted[attack])) if attack.any() else float("nan")
    bpcer = float(np.mean(~accepted[bona_fide])) if bona_fide.any() else float("nan")

    # Attack score is inverse similarity: a larger value indicates attack.
    attack_scores = 1.0 - similarities
    fpr, tpr, thresholds = roc_curve(labels, attack_scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    per_method: dict[str, dict[str, float | int]] = {}
    methods = sorted({trial.method for trial in trials if trial.is_attack})
    for method in methods:
        method_trials = [trial for trial in trials if trial.method == method]
        method_acceptance = np.asarray(
            [trial.similarity >= similarity_threshold for trial in method_trials], dtype=bool
        )
        per_method[method] = {
            "trials": len(method_trials),
            "APCER": float(np.mean(method_acceptance)) if len(method_acceptance) else float("nan"),
        }
    return {
        "threshold": float(similarity_threshold),
        "trials": len(trials),
        "attack_trials": int(np.sum(attack)),
        "bona_fide_trials": int(np.sum(bona_fide)),
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": float(np.nanmean([apcer, bpcer])),
        "attack_auc": float(roc_auc_score(labels, attack_scores)),
        "attack_eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "per_method": per_method,
    }
