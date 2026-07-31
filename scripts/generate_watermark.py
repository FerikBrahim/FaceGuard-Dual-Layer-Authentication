#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from faceguard.io import load_rgb, save_watermark_vector
from faceguard.watermark.compcode import CompCodeGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 128-bit palmprint CompCode watermark.")
    parser.add_argument("--palmprint", required=True)
    parser.add_argument("--output", default="outputs/watermark_128.json")
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    generator = CompCodeGenerator(output_bits=args.bits, seed=args.seed)
    watermark = generator.generate(load_rgb(args.palmprint))
    save_watermark_vector(args.output, watermark)
    print(f"Saved {watermark.size}-bit watermark to {Path(args.output)}")


if __name__ == "__main__":
    main()
