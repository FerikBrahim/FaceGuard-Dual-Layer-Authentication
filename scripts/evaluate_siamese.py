#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from faceguard.io import load_rgb
from faceguard.siamese.evaluate import threshold_metrics
from faceguard.siamese.model import SiameseVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Siamese checkpoint using a pairs CSV.")
    parser.add_argument("--pairs-csv", required=True, help="CSV columns: reference,query,label")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/face_metrics.json")
    parser.add_argument("--provisional-threshold", type=float, default=0.5)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs_csv)
    verifier = SiameseVerifier(
        args.checkpoint,
        threshold=args.provisional_threshold,
        trusted_checkpoint=args.trusted_checkpoint,
    )
    scores = [verifier.verify(load_rgb(row.reference), load_rgb(row.query)).similarity for row in pairs.itertuples()]
    metrics = threshold_metrics(pairs["label"].to_numpy(), np.asarray(scores))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
