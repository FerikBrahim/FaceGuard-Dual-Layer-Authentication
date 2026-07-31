#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from faceguard.siamese.deepfake import (
    DeepfakeEvalDataset,
    anti_spoofing_metrics,
    evaluate_deepfake_robustness,
    split_trials_by_identity,
)
from faceguard.siamese.model import SiameseVerifier
from faceguard.siamese.statistics import calibrate_similarity_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Identity-aware real/deepfake evaluation with APCER/BPCER.")
    parser.add_argument("--dataset-root", required=True, help="Contains real/<id> and fake/<method>/<id>")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", required=True, type=float, help="Validation-derived similarity threshold")
    parser.add_argument("--output", default="outputs/deepfake_evaluation.json")
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--calibrate-domain", action="store_true")
    parser.add_argument("--reference-frames", type=int, default=5)
    args = parser.parse_args()

    verifier = SiameseVerifier(
        args.checkpoint,
        threshold=args.threshold,
        trusted_checkpoint=args.trusted_checkpoint,
    )
    dataset = DeepfakeEvalDataset(args.dataset_root)
    trials = evaluate_deepfake_robustness(verifier, dataset, n_reference_frames=args.reference_frames)
    report: dict = {
        "original_threshold": args.threshold,
        "metrics_at_original_threshold": anti_spoofing_metrics(trials, args.threshold),
        "trials": [asdict(trial) for trial in trials],
    }
    if args.calibrate_domain:
        calibration, test = split_trials_by_identity(trials)
        # Genuine is the positive class for similarity-threshold calibration.
        labels = [int(not trial.is_attack) for trial in calibration]
        similarities = [trial.similarity for trial in calibration]
        calibrated = calibrate_similarity_threshold(labels, similarities)
        report["domain_calibration"] = calibrated
        report["metrics_at_calibrated_threshold"] = anti_spoofing_metrics(
            test, calibrated["optimal_threshold"]
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
