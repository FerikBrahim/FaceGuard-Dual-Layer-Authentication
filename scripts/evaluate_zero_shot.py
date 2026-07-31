#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from torch.utils.data import DataLoader

from faceguard.siamese.zero_shot import VerificationPairDataset, evaluate_vggface2_zero_shot


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen VGGFace2 InceptionResNetV1 baseline.")
    parser.add_argument("--pairs-csv", required=True, help="CSV columns: image_a,image_b,label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="outputs/zero_shot_metrics.json")
    args = parser.parse_args()
    with open(args.pairs_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        pairs = [(row["image_a"], row["image_b"], int(row["label"])) for row in reader]
    dataset = VerificationPairDataset(pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    metrics, _, _ = evaluate_vggface2_zero_shot(loader)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
