#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from faceguard.siamese.statistics import bootstrap_metric_ci, mcnemar_exact


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired statistical comparison of two verification models.")
    parser.add_argument("--csv", required=True, help="Columns: label,score_a,score_b")
    parser.add_argument("--threshold-a", type=float, required=True)
    parser.add_argument("--threshold-b", type=float, required=True)
    parser.add_argument("--output", default="outputs/model_comparison.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    labels = frame["label"].to_numpy(dtype=np.int32)
    score_a = frame["score_a"].to_numpy(dtype=np.float64)
    score_b = frame["score_b"].to_numpy(dtype=np.float64)
    correct_a = (score_a >= args.threshold_a).astype(np.int32) == labels
    correct_b = (score_b >= args.threshold_b).astype(np.int32) == labels
    report = {
        "mcnemar": mcnemar_exact(correct_a, correct_b),
        "model_a_accuracy_ci": bootstrap_metric_ci(
            labels, score_a, metric="accuracy", threshold=args.threshold_a
        ).__dict__,
        "model_b_accuracy_ci": bootstrap_metric_ci(
            labels, score_b, metric="accuracy", threshold=args.threshold_b
        ).__dict__,
        "model_a_auc_ci": bootstrap_metric_ci(labels, score_a, metric="auc").__dict__,
        "model_b_auc_ci": bootstrap_metric_ci(labels, score_b, metric="auc").__dict__,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
