from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image, received {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = image.astype(np.float64)
        if image.size and image.max() <= 1.0:
            image *= 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def load_rgb(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if size is not None:
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return image


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_rgb_uint8(image)).save(path)


def to_binary_vector(values: Iterable[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(list(values) if not isinstance(values, np.ndarray) else values).reshape(-1)
    if vector.size == 0:
        raise ValueError("The watermark vector is empty.")
    if np.min(vector) < 0:
        return (vector > 0).astype(np.uint8)
    return (vector >= 0.5).astype(np.uint8)


def load_watermark_vector(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return to_binary_vector(np.load(path))
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("watermark", "bits", "vector"):
                if key in data:
                    data = data[key]
                    break
        return to_binary_vector(data)
    if suffix in {".txt", ".csv"}:
        text = path.read_text(encoding="utf-8").replace(",", " ")
        return to_binary_vector([float(item) for item in text.split()])
    raise ValueError(f"Unsupported watermark file: {path}")


def save_watermark_vector(path: str | Path, vector: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vector = to_binary_vector(vector)
    if path.suffix.lower() == ".npy":
        np.save(path, vector)
    else:
        path.write_text(json.dumps({"watermark": vector.astype(int).tolist()}, indent=2), encoding="utf-8")
