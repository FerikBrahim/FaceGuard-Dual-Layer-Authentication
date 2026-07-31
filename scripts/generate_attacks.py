#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from faceguard.attacks.facial_manipulation import generate_facial_attack_suite
from faceguard.io import load_rgb, save_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled facial manipulation attacks.")
    parser.add_argument("--watermarked", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--output-dir", default="outputs/facial_attacks")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(generate_facial_attack_suite(load_rgb(args.watermarked), load_rgb(args.donor))):
        filename = f"{index:02d}_{sample.name}.png"
        save_rgb(output / filename, sample.image)
        rows.append({"file": filename, "attack": sample.name, "category": sample.category, "description": sample.description})
    pd.DataFrame(rows).to_csv(output / "attack_manifest.csv", index=False)
    print(f"Generated {len(rows)} images in {output}")


if __name__ == "__main__":
    main()
