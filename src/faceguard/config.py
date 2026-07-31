from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WatermarkConfig:
    length: int = 128
    working_size: int = 512
    margin: float = 36.0
    block_size: int = 4
    repetitions: int = 5
    level: int = 2
    orientation_index: int = 2
    svd_index: int = 0
    seed: int = 2026
    ncc_threshold: float = 0.90
    ber_threshold: float = 0.05


@dataclass
class SiameseConfig:
    face_size: int = 160
    embedding_dim: int = 512
    min_face_probability: float = 0.90
    threshold: float | None = None
    margin: float = 0.8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 15
    seed: int = 2026


@dataclass
class ExperimentConfig:
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    siamese: SiameseConfig = field(default_factory=SiameseConfig)
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(
            watermark=WatermarkConfig(**raw.get("watermark", {})),
            siamese=SiameseConfig(**raw.get("siamese", {})),
            output_dir=raw.get("output_dir", "outputs"),
        )
