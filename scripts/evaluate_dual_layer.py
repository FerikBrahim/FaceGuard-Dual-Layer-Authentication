#!/usr/bin/env python3
from __future__ import annotations

import argparse

from faceguard.pipeline import evaluate_dual_layer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FaceGuard under facial manipulation attacks.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--watermarked", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--siamese-threshold", required=True, type=float)
    parser.add_argument("--watermark")
    parser.add_argument("--key")
    parser.add_argument("--output-dir", default="outputs/dual_layer")
    parser.add_argument("--trusted-checkpoint", action="store_true")
    args = parser.parse_args()
    if bool(args.watermark) != bool(args.key):
        parser.error("--watermark and --key must be supplied together.")
    results = evaluate_dual_layer(
        reference_path=args.reference,
        watermarked_path=args.watermarked,
        donor_path=args.donor,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        siamese_threshold=args.siamese_threshold,
        watermark_vector_path=args.watermark,
        watermark_key_path=args.key,
        trusted_checkpoint=args.trusted_checkpoint,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
