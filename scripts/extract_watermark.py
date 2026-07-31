#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from faceguard.io import load_rgb, load_watermark_vector, save_watermark_vector
from faceguard.metrics import watermark_metrics
from faceguard.watermark.dtcwt_svd import PairwiseDTCWTSVDWatermarker, PairwiseSVDKey


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and optionally verify a FaceGuard watermark.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--original-watermark")
    parser.add_argument("--output", default="outputs/extracted_watermark.json")
    parser.add_argument("--ncc-threshold", type=float, default=0.90)
    parser.add_argument("--ber-threshold", type=float, default=0.05)
    args = parser.parse_args()
    key = PairwiseSVDKey.from_json(args.key)
    system = PairwiseDTCWTSVDWatermarker(
        margin=key.margin,
        block_size=key.block_size,
        repetitions=key.repetitions,
        level=key.level,
        orientation_index=key.orientation_index,
        svd_index=key.svd_index,
        seed=key.seed,
        watermark_length=key.watermark_length,
        working_size=key.working_shape[0],
    )
    extracted = system.extract(load_rgb(args.image), key)
    save_watermark_vector(args.output, extracted)
    if args.original_watermark:
        metrics = watermark_metrics(
            load_watermark_vector(args.original_watermark),
            extracted,
            args.ncc_threshold,
            args.ber_threshold,
        )
        print(json.dumps(metrics, indent=2))
    print(f"Saved extracted watermark to {Path(args.output)}")


if __name__ == "__main__":
    main()
