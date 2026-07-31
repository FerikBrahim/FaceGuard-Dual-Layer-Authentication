from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from faceguard.attacks.facial_manipulation import generate_facial_attack_suite
from faceguard.fusion.authentication import fuse_decisions
from faceguard.io import load_rgb, save_rgb
from faceguard.metrics import image_quality, watermark_metrics
from faceguard.siamese.model import SiameseVerifier
from faceguard.watermark.dtcwt_svd import PairwiseDTCWTSVDWatermarker, PairwiseSVDKey


def evaluate_dual_layer(
    reference_path: str | Path,
    watermarked_path: str | Path,
    donor_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    siamese_threshold: float,
    watermark_vector_path: str | Path | None = None,
    watermark_key_path: str | Path | None = None,
    trusted_checkpoint: bool = False,
    ncc_threshold: float = 0.90,
    ber_threshold: float = 0.05,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    reference = load_rgb(reference_path, (512, 512))
    watermarked = load_rgb(watermarked_path, None)
    donor = load_rgb(donor_path, (512, 512))
    verifier = SiameseVerifier(
        checkpoint_path=checkpoint_path,
        threshold=siamese_threshold,
        trusted_checkpoint=trusted_checkpoint,
    )
    watermark_system = None
    key = None
    original_watermark = None
    if watermark_vector_path is not None and watermark_key_path is not None:
        from faceguard.io import load_watermark_vector

        original_watermark = load_watermark_vector(watermark_vector_path)
        key = PairwiseSVDKey.from_json(watermark_key_path)
        watermark_system = PairwiseDTCWTSVDWatermarker(
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
    rows: list[dict[str, Any]] = []
    for index, attack in enumerate(generate_facial_attack_suite(watermarked, donor)):
        save_rgb(image_dir / f"{index:02d}_{attack.name}.png", attack.image)
        siamese = verifier.verify(reference, attack.image)
        quality = image_quality(watermarked if attack.category != "control" else reference, attack.image)
        wm = None
        if watermark_system is not None and key is not None and original_watermark is not None:
            extracted = watermark_system.extract(attack.image, key)
            wm = watermark_metrics(original_watermark, extracted, ncc_threshold, ber_threshold)
        fusion = fuse_decisions(siamese.accepted, None if wm is None else bool(wm["Watermark_Accepted"]))
        rows.append(
            {
                "Query condition": attack.name,
                "Attack category": attack.category,
                "Cosine similarity s_s": siamese.similarity,
                "A_s": fusion.A_s,
                "PSNR": quality["PSNR"],
                "SSIM": quality["SSIM"],
                "MSE": quality["MSE"],
                "NCC": np.nan if wm is None else wm["NCC"],
                "BER (%)": np.nan if wm is None else wm["BER_percent"],
                "A_w": np.nan if fusion.A_w is None else fusion.A_w,
                "A_f": np.nan if fusion.A_f is None else fusion.A_f,
                "Final outcome": fusion.outcome.value,
            }
        )
    dataframe = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_dir / "facial_manipulation_results.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "watermark_branch_measured": watermark_system is not None,
                "siamese_threshold": siamese_threshold,
                "ncc_threshold": ncc_threshold,
                "ber_threshold": ber_threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataframe
