#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from faceguard.io import load_rgb, load_watermark_vector, save_rgb
from faceguard.metrics import image_quality
from faceguard.watermark.dtcwt_svd import PairwiseDTCWTSVDWatermarker

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-watermark labelled face images.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--watermark", required=True, help="Shared 128-bit vector for this experiment")
    parser.add_argument("--margin", type=float, default=36.0)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    input_root, output_root = Path(args.input_root), Path(args.output_root)
    watermark = load_watermark_vector(args.watermark)
    system = PairwiseDTCWTSVDWatermarker(
        margin=args.margin,
        block_size=args.block_size,
        repetitions=args.repetitions,
        seed=args.seed,
        watermark_length=len(watermark),
    )
    rows = []
    for path in sorted(p for p in input_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        relative = path.relative_to(input_root)
        output_image = output_root / relative.with_suffix(".png")
        output_key = output_root / "keys" / relative.with_suffix(".json")
        image = load_rgb(path, (512, 512))
        watermarked, key = system.embed(image, watermark)
        save_rgb(output_image, watermarked)
        key.to_json(output_key)
        rows.append({"source": str(relative), "output": str(output_image), **image_quality(image, watermarked)})
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_root / "imperceptibility_results.csv", index=False)
    (output_root / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Processed {len(rows)} images; results written to {output_root}")


if __name__ == "__main__":
    main()
