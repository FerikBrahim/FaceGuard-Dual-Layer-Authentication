from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from faceguard.io import ensure_rgb_uint8


@dataclass
class RobustnessAttack:
    name: str
    image: np.ndarray


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(ensure_rgb_uint8(image), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def gaussian_noise(image: np.ndarray, sigma: float, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = ensure_rgb_uint8(image).astype(np.float32)
    return np.clip(values + rng.normal(0.0, sigma, values.shape), 0, 255).astype(np.uint8)


def salt_pepper(image: np.ndarray, amount: float, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = ensure_rgb_uint8(image).copy()
    count = int(amount * output.shape[0] * output.shape[1])
    coordinates = (rng.integers(0, output.shape[0], count), rng.integers(0, output.shape[1], count))
    output[coordinates] = rng.choice([0, 255], size=(count, 1))
    return output


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    image = ensure_rgb_uint8(image)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def resize_restore(image: np.ndarray, scale: float) -> np.ndarray:
    image = ensure_rgb_uint8(image)
    height, width = image.shape[:2]
    reduced = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)


def crop_restore(image: np.ndarray, retained: float) -> np.ndarray:
    image = ensure_rgb_uint8(image)
    height, width = image.shape[:2]
    crop_h, crop_w = int(height * retained), int(width * retained)
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    return cv2.resize(image[top : top + crop_h, left : left + crop_w], (width, height), interpolation=cv2.INTER_CUBIC)


def generate_robustness_suite(image: np.ndarray) -> list[RobustnessAttack]:
    source = ensure_rgb_uint8(image)
    return [
        RobustnessAttack("no_attack", source.copy()),
        RobustnessAttack("jpeg_q90", jpeg(source, 90)),
        RobustnessAttack("jpeg_q70", jpeg(source, 70)),
        RobustnessAttack("jpeg_q50", jpeg(source, 50)),
        RobustnessAttack("jpeg_q30", jpeg(source, 30)),
        RobustnessAttack("gaussian_noise_sigma3", gaussian_noise(source, 3)),
        RobustnessAttack("gaussian_noise_sigma5", gaussian_noise(source, 5)),
        RobustnessAttack("salt_pepper_0005", salt_pepper(source, 0.005)),
        RobustnessAttack("salt_pepper_001", salt_pepper(source, 0.01)),
        RobustnessAttack("median_filter_3", cv2.medianBlur(source, 3)),
        RobustnessAttack("gaussian_blur_3", cv2.GaussianBlur(source, (3, 3), 0)),
        RobustnessAttack("resize_75", resize_restore(source, 0.75)),
        RobustnessAttack("resize_50", resize_restore(source, 0.50)),
        RobustnessAttack("rotation_2", rotate(source, 2)),
        RobustnessAttack("rotation_5", rotate(source, 5)),
        RobustnessAttack("crop_90", crop_restore(source, 0.90)),
        RobustnessAttack("crop_80", crop_restore(source, 0.80)),
    ]
