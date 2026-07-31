#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from faceguard.io import load_rgb, load_watermark_vector, save_rgb
from faceguard.metrics import image_quality
from faceguard.watermark.dtcwt_svd import PairwiseDTCWTSVDWatermarker


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a binary watermark into a face image.")
    parser.add_argument("--face", required=True)
    parser.add_argument("--watermark", required=True)
    parser.add_argument("--output", default="outputs/watermarked_face.png")
    parser.add_argument("--key", default="outputs/watermark_key.json")
    parser.add_argument("--margin", type=float, default=36.0)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    face = load_rgb(args.face)
    watermark = load_watermark_vector(args.watermark)
    system = PairwiseDTCWTSVDWatermarker(
        margin=args.margin,
        block_size=args.block_size,
        repetitions=args.repetitions,
        seed=args.seed,
        watermark_length=watermark.size,
    )
    watermarked, key = system.embed(face, watermark)
    save_rgb(args.output, watermarked)
    key.to_json(args.key)
    print(image_quality(face, watermarked))
    print(f"Saved image to {Path(args.output)} and key to {Path(args.key)}")


if __name__ == "__main__":
    main()
