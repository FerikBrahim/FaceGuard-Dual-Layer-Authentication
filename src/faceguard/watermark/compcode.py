from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from faceguard.io import ensure_rgb_uint8


@dataclass
class CompCodeGenerator:
    """Generate a fixed-length palmprint CompCode watermark.

    The implementation follows the manuscript pipeline: six directional Gabor
    responses, winner-take-all orientation coding, block histograms, binary
    quantization, and deterministic selection of a fixed-length bit vector.
    """

    output_bits: int = 128
    image_size: int = 256
    block_size: int = 16
    seed: int = 2026
    orientations: tuple[int, ...] = (0, 30, 60, 90, 120, 150)

    def _kernel(self, angle_degrees: float) -> np.ndarray:
        return cv2.getGaborKernel(
            ksize=(31, 31),
            sigma=5.0,
            theta=np.deg2rad(angle_degrees),
            lambd=8.0,
            gamma=0.5,
            psi=0.0,
            ktype=cv2.CV_32F,
        )

    def orientation_map(self, palmprint_rgb: np.ndarray) -> np.ndarray:
        image = ensure_rgb_uint8(palmprint_rgb)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        responses = [
            np.abs(cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, self._kernel(angle)))
            for angle in self.orientations
        ]
        return np.argmax(np.stack(responses, axis=0), axis=0).astype(np.uint8)

    def feature_vector(self, palmprint_rgb: np.ndarray) -> np.ndarray:
        code = self.orientation_map(palmprint_rgb)
        features: list[float] = []
        for row in range(0, code.shape[0], self.block_size):
            for col in range(0, code.shape[1], self.block_size):
                block = code[row : row + self.block_size, col : col + self.block_size]
                hist = np.bincount(block.reshape(-1), minlength=len(self.orientations)).astype(np.float64)
                hist /= max(float(hist.sum()), 1.0)
                features.extend(hist.tolist())
        return np.asarray(features, dtype=np.float64)

    def generate(self, palmprint_rgb: np.ndarray) -> np.ndarray:
        features = self.feature_vector(palmprint_rgb)
        threshold = float(np.median(features))
        candidate_bits = (features >= threshold).astype(np.uint8)
        if candidate_bits.size < self.output_bits:
            repeats = int(np.ceil(self.output_bits / candidate_bits.size))
            candidate_bits = np.tile(candidate_bits, repeats)
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(candidate_bits.size, size=self.output_bits, replace=False)
        return candidate_bits[indices].astype(np.uint8)
