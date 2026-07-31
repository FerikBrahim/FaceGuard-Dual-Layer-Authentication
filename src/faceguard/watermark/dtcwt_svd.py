from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from faceguard.io import ensure_rgb_uint8, to_binary_vector

# Compatibility aliases required by dtcwt 0.14 on newer NumPy releases.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda value, dtype=np.float64: np.asarray(value, dtype=dtype)  # type: ignore[attr-defined]
if not hasattr(np, "issubsctype"):
    np.issubsctype = lambda left, right: np.issubdtype(np.asarray(left).dtype, right)  # type: ignore[attr-defined]

try:
    import dtcwt
    from dtcwt.numpy.common import Pyramid
except ImportError as exc:  # pragma: no cover - optional dependency message
    dtcwt = None
    Pyramid = None
    _DTCWT_IMPORT_ERROR = exc
else:
    _DTCWT_IMPORT_ERROR = None


@dataclass
class PairwiseSVDKey:
    seed: int
    watermark_length: int
    block_size: int
    level: int
    orientation_index: int
    margin: float
    repetitions: int
    svd_index: int
    original_shape: list[int]
    working_shape: list[int]
    pairs: list[list[int]]
    spread_mask: list[int]

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "PairwiseSVDKey":
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in allowed}
        missing = sorted(allowed - set(filtered))
        if missing:
            raise ValueError(f"Watermark key is missing fields: {', '.join(missing)}")
        return cls(**filtered)


class PairwiseDTCWTSVDWatermarker:
    """Blind pairwise DT-CWT–SVD watermarking with spread masking.

    A bit is represented by the ordering of one selected singular value in two
    DT-CWT blocks. Extraction requires the JSON key but not the original host.
    """

    def __init__(
        self,
        margin: float = 36.0,
        block_size: int = 4,
        repetitions: int = 5,
        level: int = 2,
        orientation_index: int = 2,
        svd_index: int = 0,
        seed: int = 2026,
        watermark_length: int = 128,
        working_size: int = 512,
    ) -> None:
        if dtcwt is None:
            raise ImportError("Install dtcwt>=0.14 to use the watermarking module.") from _DTCWT_IMPORT_ERROR
        self.margin = float(margin)
        self.block_size = int(block_size)
        self.repetitions = int(repetitions)
        self.level = int(level)
        self.orientation_index = int(orientation_index)
        self.svd_index = int(svd_index)
        self.seed = int(seed)
        self.watermark_length = int(watermark_length)
        self.working_size = int(working_size)
        self.transform = dtcwt.Transform2d()

    def _preprocess(self, image_rgb: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
        image = ensure_rgb_uint8(image_rgb)
        original_shape = list(image.shape)
        image = cv2.resize(image, (self.working_size, self.working_size), interpolation=cv2.INTER_CUBIC)
        return image, original_shape, list(image.shape)

    def _forward(self, luminance: np.ndarray):
        pyramid = self.transform.forward(luminance.astype(np.float64), nlevels=self.level)
        highpasses = [plane.copy() for plane in pyramid.highpasses]
        selected = highpasses[self.level - 1][:, :, self.orientation_index]
        return pyramid, highpasses, np.abs(selected), np.exp(1j * np.angle(selected))

    def _inverse(self, pyramid, highpasses, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        highpasses[self.level - 1][:, :, self.orientation_index] = magnitude * phase
        try:
            rebuilt = Pyramid(pyramid.lowpass, tuple(highpasses), pyramid.scales)
        except TypeError:
            rebuilt = Pyramid(pyramid.lowpass, tuple(highpasses), scales=pyramid.scales)
        return np.clip(self.transform.inverse(rebuilt), 0, 255).astype(np.uint8)

    def _positions(self, shape: tuple[int, int]) -> list[tuple[int, int]]:
        height, width = shape
        size = self.block_size
        return [
            (row, col)
            for row in range(0, height - size + 1, size)
            for col in range(0, width - size + 1, size)
        ]

    def _block(self, plane: np.ndarray, position: tuple[int, int]) -> np.ndarray:
        row, col = position
        return plane[row : row + self.block_size, col : col + self.block_size]

    def _set_block(self, plane: np.ndarray, position: tuple[int, int], block: np.ndarray) -> None:
        row, col = position
        plane[row : row + self.block_size, col : col + self.block_size] = block

    def _sigma(self, block: np.ndarray) -> float:
        singular = np.linalg.svd(block.astype(np.float64), compute_uv=False)
        return float(singular[self.svd_index])

    def _replace_sigma(self, block: np.ndarray, value: float) -> np.ndarray:
        left, singular, right = np.linalg.svd(block.astype(np.float64), full_matrices=False)
        singular[self.svd_index] = max(float(value), 1e-6)
        return left @ np.diag(singular) @ right

    def capacity_bits(self, image_rgb: np.ndarray) -> int:
        image, _, _ = self._preprocess(image_rgb)
        ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        _, _, magnitude, _ = self._forward(ycc[:, :, 0])
        return (len(self._positions(magnitude.shape)) // 2) // self.repetitions

    def embed(self, image_rgb: np.ndarray, watermark: np.ndarray) -> tuple[np.ndarray, PairwiseSVDKey]:
        image, original_shape, working_shape = self._preprocess(image_rgb)
        bits = to_binary_vector(watermark)
        if bits.size != self.watermark_length:
            raise ValueError(f"Expected {self.watermark_length} watermark bits, received {bits.size}.")
        ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        pyramid, highpasses, magnitude, phase = self._forward(ycc[:, :, 0])
        plane = magnitude.copy()
        positions = self._positions(plane.shape)
        required = bits.size * self.repetitions * 2
        if required > len(positions):
            raise ValueError(
                f"Insufficient capacity: need {required} blocks, available {len(positions)}. "
                "Reduce repetitions or block size."
            )
        rng = np.random.default_rng(self.seed)
        selected = [positions[index] for index in rng.permutation(len(positions))[:required]]
        mask = rng.integers(0, 2, size=bits.size, dtype=np.uint8)
        spread = np.bitwise_xor(bits, mask)
        pairs: list[list[int]] = []
        cursor = 0
        for bit in spread:
            for _ in range(self.repetitions):
                first, second = selected[cursor], selected[cursor + 1]
                cursor += 2
                block_a, block_b = self._block(plane, first), self._block(plane, second)
                sigma_a, sigma_b = self._sigma(block_a), self._sigma(block_b)
                center = 0.5 * (sigma_a + sigma_b)
                sign = 1.0 if bit == 1 else -1.0
                target_a = center + sign * self.margin / 2.0
                target_b = center - sign * self.margin / 2.0
                self._set_block(plane, first, self._replace_sigma(block_a, target_a))
                self._set_block(plane, second, self._replace_sigma(block_b, target_b))
                pairs.append([first[0], first[1], second[0], second[1]])
        reconstructed_y = self._inverse(pyramid, highpasses, np.maximum(plane, 0.0), phase)
        ycc[:, :, 0] = reconstructed_y
        watermarked = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
        if list(watermarked.shape) != original_shape:
            watermarked = cv2.resize(watermarked, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        key = PairwiseSVDKey(
            seed=self.seed,
            watermark_length=bits.size,
            block_size=self.block_size,
            level=self.level,
            orientation_index=self.orientation_index,
            margin=self.margin,
            repetitions=self.repetitions,
            svd_index=self.svd_index,
            original_shape=original_shape,
            working_shape=working_shape,
            pairs=pairs,
            spread_mask=mask.astype(int).tolist(),
        )
        return ensure_rgb_uint8(watermarked), key

    def extract(self, suspected_rgb: np.ndarray, key: PairwiseSVDKey) -> np.ndarray:
        image = ensure_rgb_uint8(suspected_rgb)
        height, width = key.working_shape[:2]
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        _, _, magnitude, _ = self._forward(ycc[:, :, 0])
        raw: list[int] = []
        for row_a, col_a, row_b, col_b in key.pairs:
            sigma_a = self._sigma(self._block(magnitude, (int(row_a), int(col_a))))
            sigma_b = self._sigma(self._block(magnitude, (int(row_b), int(col_b))))
            raw.append(1 if sigma_a > sigma_b else 0)
        repeated = np.asarray(raw, dtype=np.uint8).reshape(key.watermark_length, key.repetitions)
        spread = (np.mean(repeated, axis=1) >= 0.5).astype(np.uint8)
        return np.bitwise_xor(spread, np.asarray(key.spread_mask, dtype=np.uint8))
