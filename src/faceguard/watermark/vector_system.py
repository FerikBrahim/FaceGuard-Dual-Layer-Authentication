# Colab-ready module for vector biometric watermarking.
# In Colab, run:
# !pip -q install numpy opencv-python-headless scikit-image pandas openpyxl dtcwt PyWavelets
# Then upload this file or run the notebook watermark_vector_system_colab.ipynb.

"""
Vector Biometric Watermarking System for Face Images
====================================================

Implements the watermark part of a Siamese Network-Based Biometric Watermarking paper:
    - direct vector watermark input, no palmprint generation stage required
    - transform-domain embedding in the luminance channel
    - DT-CWT + SVD blind QIM embedding/extraction
    - imperceptibility metrics: MSE, PSNR, SSIM
    - watermark metrics: NCC, BER, extraction accuracy
    - robustness tests under common image attacks
    - capacity report in bits, bpp, and redundancy factor
    - key sensitivity / wrong-key verification helper

Main idea
---------
The paper's additive SVD formula, Sigma' = Sigma + alpha*m(W'), is difficult to
extract blindly because the original singular values are unknown. This code uses
QIM/parity quantization of a selected singular value. This keeps the same spirit
of SVD-domain watermarking, but makes direct extraction from the suspected image
possible using only the secret key and parameters.

Dependencies
------------
Required:
    pip install numpy opencv-python scikit-image pandas

For true DT-CWT as described in the paper:
    pip install dtcwt

Optional fallback DWT mode:
    pip install PyWavelets

Example
-------
    python watermark_vector_system.py \
        --face face.png \
        --watermark watermark.npy \
        --out_dir results \
        --delta 4.0 \
        --seed 2026 \
        --transform dtcwt

If no watermark file is provided, a random binary vector is generated for testing:
    python watermark_vector_system.py --face face.png --demo_wm_len 512

Provenance: refactored from the authors' Colab vector-watermarking notebook.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Colab / NumPy 2.x compatibility
# -----------------------------------------------------------------------------
# The dtcwt package still calls np.asfarray in some versions. NumPy 2.0 removed
# np.asfarray, so we provide a small compatibility alias before importing dtcwt.
if not hasattr(np, "asfarray"):
    def _np_asfarray_compat(a, dtype=np.float64):
        return np.asarray(a, dtype=dtype)
    np.asfarray = _np_asfarray_compat  # type: ignore[attr-defined]

# Some dtcwt versions also call np.issubsctype, removed in NumPy 2.0.
# Equivalent behavior can be obtained with np.issubdtype after resolving
# arrays/scalars/classes to a dtype. This patch must run before dtcwt is used.
if not hasattr(np, "issubsctype"):
    def _np_issubsctype_compat(arg1, arg2):
        def _resolve_dtype(x):
            if isinstance(x, np.ndarray):
                return x.dtype
            try:
                return np.dtype(x)
            except TypeError:
                return np.asarray(x).dtype
        return np.issubdtype(_resolve_dtype(arg1), arg2)
    np.issubsctype = _np_issubsctype_compat  # type: ignore[attr-defined]

MIN_WATERMARK_BITS = 128

import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8."""
    path = str(path)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def save_image_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    """Save an RGB image."""
    image_rgb = np.asarray(image_rgb)
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(np.rint(image_rgb), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def rgb_to_ycrcb_float(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB uint8/float image to OpenCV YCrCb float64.

    Note: OpenCV uses YCrCb ordering. We keep Cb/Cr unchanged and only modify Y.
    This is equivalent for our purpose because the same inverse conversion is used.
    """
    image_rgb_u8 = np.clip(np.rint(image_rgb), 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    return ycrcb


def ycrcb_float_to_rgb(ycrcb: np.ndarray) -> np.ndarray:
    ycrcb_u8 = np.clip(np.rint(ycrcb), 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(ycrcb_u8, cv2.COLOR_YCrCb2RGB)
    return rgb


def normalize_watermark_vector(w: np.ndarray | Sequence[Any]) -> np.ndarray:
    """
    Convert a watermark vector to binary {0,1} int array.

    Accepted inputs:
        - already binary: 0/1
        - bipolar: -1/+1
        - floats: values > 0 become 1, values <= 0 become 0
    """
    arr = np.asarray(w).reshape(-1)
    if arr.size == 0:
        raise ValueError("Watermark vector is empty.")
    unique = set(np.unique(arr).tolist())
    if unique.issubset({0, 1, 0.0, 1.0, False, True}):
        bits = arr.astype(np.uint8)
    elif unique.issubset({-1, 1, -1.0, 1.0}):
        bits = (arr > 0).astype(np.uint8)
    else:
        bits = (arr > 0).astype(np.uint8)
    return bits.reshape(-1)




def enforce_minimum_watermark_length(
    bits: np.ndarray | Sequence[Any],
    min_bits: int = MIN_WATERMARK_BITS,
    mode: str = "repeat",
) -> np.ndarray:
    """
    Ensure that a binary watermark vector has at least ``min_bits`` bits.

    If the input vector is shorter than ``min_bits``, the default behavior is to
    repeat it until the minimum length is reached, then truncate to exactly
    ``min_bits``. For example, a 32-bit vector becomes a 128-bit vector by
    repeating it four times.

    Notes for research use:
        - Repeating a 32-bit vector is convenient for testing.
        - For final experiments, prefer generating a true biometric vector of
          128 bits or more to increase entropy and security.
    """
    arr = normalize_watermark_vector(bits)
    min_bits = int(min_bits)
    if min_bits <= 0:
        raise ValueError("min_bits must be positive")
    if arr.size >= min_bits:
        return arr.astype(np.uint8)
    if mode.lower() == "repeat":
        reps = int(np.ceil(min_bits / arr.size))
        expanded = np.tile(arr, reps)[:min_bits]
    elif mode.lower() == "pad_zero":
        expanded = np.zeros(min_bits, dtype=np.uint8)
        expanded[: arr.size] = arr
    else:
        raise ValueError("mode must be 'repeat' or 'pad_zero'")
    print(
        f"Watermark vector was {arr.size} bits; expanded to {expanded.size} bits "
        f"using mode='{mode}'."
    )
    return expanded.astype(np.uint8)


def load_watermark_vector(path: str | Path) -> np.ndarray:
    """Load watermark vector from .npy, .txt, .csv, or .json."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Watermark vector not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        vec = np.load(path)
    elif suffix in {".txt", ".csv"}:
        vec = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            vec = np.asarray(json.load(f))
    else:
        raise ValueError("Unsupported watermark file. Use .npy, .txt, .csv, or .json")
    return normalize_watermark_vector(vec)


def save_watermark_vector(path: str | Path, bits: np.ndarray) -> None:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        np.save(path, bits.astype(np.uint8))
    elif path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bits.astype(int).tolist(), f)
    else:
        np.savetxt(path, bits.astype(int), fmt="%d")


def binary_ncc(w: np.ndarray, w_hat: np.ndarray) -> float:
    """
    Normalized cross-correlation for binary watermark vectors.
    Binary values are mapped to {-1,+1} to avoid bias from many zeros.
    """
    a = 2 * normalize_watermark_vector(w).astype(np.float64) - 1.0
    b = 2 * normalize_watermark_vector(w_hat).astype(np.float64) - 1.0
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    a = a[:n]
    b = b[:n]
    denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def bit_error_rate(w: np.ndarray, w_hat: np.ndarray) -> float:
    a = normalize_watermark_vector(w)
    b = normalize_watermark_vector(w_hat)
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float(np.mean(a[:n] != b[:n]))


def image_imperceptibility_metrics(original_rgb: np.ndarray, watermarked_rgb: np.ndarray) -> Dict[str, float]:
    original = np.clip(original_rgb, 0, 255).astype(np.uint8)
    watermarked = np.clip(watermarked_rgb, 0, 255).astype(np.uint8)
    mse = float(np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2))
    psnr = float(peak_signal_noise_ratio(original, watermarked, data_range=255))
    try:
        ssim = float(structural_similarity(original, watermarked, channel_axis=2, data_range=255))
    except TypeError:
        # older skimage compatibility
        ssim = float(structural_similarity(original, watermarked, multichannel=True, data_range=255))
    return {"MSE": mse, "PSNR_dB": psnr, "SSIM": ssim}


def watermark_similarity_metrics(w: np.ndarray, w_hat: np.ndarray) -> Dict[str, float]:
    ber = bit_error_rate(w, w_hat)
    return {
        "NCC": binary_ncc(w, w_hat),
        "BER": ber,
        "BER_percent": 100.0 * ber,
        "Extraction_Accuracy": 1.0 - ber,
        "Extraction_Accuracy_percent": 100.0 * (1.0 - ber),
    }


# -----------------------------------------------------------------------------
# Key/config dataclass
# -----------------------------------------------------------------------------


@dataclass
class WatermarkKey:
    transform: str
    seed: int
    watermark_length: int
    original_image_shape: Tuple[int, int, int]
    delta: float
    block_size: int
    svd_index: int
    level: int
    orientation_index: int
    highpass_shape: Tuple[int, int]
    capacity_bits: int
    used_blocks: int
    repetition: bool
    wavelet: str = "haar"

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str | Path) -> "WatermarkKey":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["original_image_shape"] = tuple(d["original_image_shape"])
        d["highpass_shape"] = tuple(d["highpass_shape"])
        return WatermarkKey(**d)


# -----------------------------------------------------------------------------
# Transform backends: DT-CWT and optional DWT fallback
# -----------------------------------------------------------------------------


class TransformBackend:
    def __init__(self, transform: str = "dtcwt", level: int = 2, wavelet: str = "haar") -> None:
        self.transform = transform.lower()
        self.level = level
        self.wavelet = wavelet
        self._dtcwt_transform = None
        if self.transform == "dtcwt":
            try:
                from dtcwt.numpy import Transform2d  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "The 'dtcwt' package is required for transform='dtcwt'. "
                    "Install it with: pip install dtcwt. "
                    "Alternatively, use --transform dwt as a fallback for testing."
                ) from exc
            self._dtcwt_transform = Transform2d()
        elif self.transform == "dwt":
            try:
                import pywt  # noqa: F401
            except Exception as exc:
                raise ImportError(
                    "PyWavelets is required for transform='dwt'. Install it with: pip install PyWavelets"
                ) from exc
        else:
            raise ValueError("transform must be either 'dtcwt' or 'dwt'")

    def forward(self, y: np.ndarray) -> Any:
        y = np.asarray(y, dtype=np.float64)
        if self.transform == "dtcwt":
            assert self._dtcwt_transform is not None
            return self._dtcwt_transform.forward(y, nlevels=self.level)
        import pywt
        return pywt.wavedec2(y, wavelet=self.wavelet, level=self.level, mode="periodization")

    def get_selected_subband(self, coeffs: Any, orientation_index: int) -> np.ndarray:
        if self.transform == "dtcwt":
            # highpasses[level-1] has shape (rows, cols, 6), complex-valued.
            high = coeffs.highpasses[self.level - 1]
            if orientation_index < 0 or orientation_index >= high.shape[2]:
                raise ValueError(f"DT-CWT orientation_index must be 0..{high.shape[2]-1}")
            return high[:, :, orientation_index]
        # DWT fallback: coeffs = [cA2, (cH2,cV2,cD2), (cH1,cV1,cD1)]
        # Use cV at requested level as approximate vertical/HL-like detail.
        detail_tuple = coeffs[1]  # level-2 details when level=2
        cH, cV, cD = detail_tuple
        if orientation_index == 0:
            return cH
        if orientation_index == 1:
            return cV
        return cD

    def replace_selected_subband(self, coeffs: Any, new_subband: np.ndarray, orientation_index: int, original_shape: Tuple[int, int]) -> np.ndarray:
        if self.transform == "dtcwt":
            assert self._dtcwt_transform is not None
            highpasses = list(coeffs.highpasses)
            high = np.array(highpasses[self.level - 1], copy=True)
            high[:, :, orientation_index] = new_subband
            highpasses[self.level - 1] = high
            try:
                from dtcwt.numpy.common import Pyramid  # type: ignore
                try:
                    new_coeffs = Pyramid(coeffs.lowpass, tuple(highpasses), coeffs.scales)
                except TypeError:
                    new_coeffs = Pyramid(coeffs.lowpass, tuple(highpasses))
            except Exception:
                # Compatibility fallback: some versions expose Pyramid at top level.
                import dtcwt  # type: ignore
                try:
                    new_coeffs = dtcwt.Pyramid(coeffs.lowpass, tuple(highpasses), coeffs.scales)
                except TypeError:
                    new_coeffs = dtcwt.Pyramid(coeffs.lowpass, tuple(highpasses))
            rec = self._dtcwt_transform.inverse(new_coeffs)
            return np.asarray(rec[: original_shape[0], : original_shape[1]], dtype=np.float64)

        import pywt
        coeffs_new = list(coeffs)
        cH, cV, cD = coeffs_new[1]
        if orientation_index == 0:
            coeffs_new[1] = (new_subband, cV, cD)
        elif orientation_index == 1:
            coeffs_new[1] = (cH, new_subband, cD)
        else:
            coeffs_new[1] = (cH, cV, new_subband)
        rec = pywt.waverec2(coeffs_new, wavelet=self.wavelet, mode="periodization")
        return np.asarray(rec[: original_shape[0], : original_shape[1]], dtype=np.float64)


# -----------------------------------------------------------------------------
# Core watermarking class
# -----------------------------------------------------------------------------


class VectorWatermarkSystem:
    """
    Blind vector watermark embedding/extraction using transform-domain SVD-QIM.

    Parameters
    ----------
    delta:
        QIM quantization step. In the paper, this corresponds to the embedding
        strength alpha. Larger delta means stronger/more robust embedding, but lower
        imperceptibility.
    block_size:
        Non-overlapping SVD block size in the selected level-2 subband.
    seed:
        Secret seed for pseudo-random spreading.
    transform:
        'dtcwt' for paper-compatible DT-CWT, or 'dwt' fallback for testing.
    level:
        Decomposition level. The paper uses level=2.
    orientation_index:
        DT-CWT has six orientations at each level. Use 0..5. Default 2.
        In DWT fallback: 0=cH, 1=cV, 2=cD.
    svd_index:
        Which singular value to quantize. 0 is strongest and generally robust.
    repetition:
        If True, repeat the watermark bits over all available blocks and use majority
        voting at extraction. This improves robustness and still recovers the original
        watermark vector length. If False, embed exactly one bit per block for the
        first watermark_length blocks.
    """

    def __init__(
        self,
        delta: float = 4.0,
        block_size: int = 8,
        seed: int = 2026,
        transform: str = "dtcwt",
        level: int = 2,
        orientation_index: int = 2,
        svd_index: int = 0,
        repetition: bool = True,
        wavelet: str = "haar",
    ) -> None:
        if delta <= 0:
            raise ValueError("delta must be > 0")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        if svd_index < 0 or svd_index >= block_size:
            raise ValueError("svd_index must be in [0, block_size-1]")
        self.delta = float(delta)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.transform_name = transform.lower()
        self.level = int(level)
        self.orientation_index = int(orientation_index)
        self.svd_index = int(svd_index)
        self.repetition = bool(repetition)
        self.wavelet = wavelet
        self.backend = TransformBackend(transform=self.transform_name, level=self.level, wavelet=self.wavelet)

    # ---- QIM functions -------------------------------------------------------

    @staticmethod
    def _qim_embed_scalar(s: float, bit: int, delta: float) -> float:
        """Embed one bit by forcing round(s/delta) parity to bit."""
        bit = int(bit) & 1
        q_center = int(np.round(s / delta))
        candidates = []
        for q in range(q_center - 4, q_center + 5):
            if q >= 0 and (q % 2) == bit:
                candidates.append(q)
        if not candidates:
            candidates = [bit]
        best_q = min(candidates, key=lambda q: abs((q * delta) - s))
        return float(best_q * delta)

    @staticmethod
    def _qim_extract_scalar(s: float, delta: float) -> int:
        q = int(np.round(s / delta))
        return q & 1

    # ---- block helpers -------------------------------------------------------

    def _block_positions(self, subband_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        h, w = subband_shape
        bs = self.block_size
        return [(r, c) for r in range(0, h - bs + 1, bs) for c in range(0, w - bs + 1, bs)]

    def _make_schedule_and_prn(self, wm_len: int, capacity: int) -> Tuple[np.ndarray, np.ndarray, int]:
        if wm_len <= 0:
            raise ValueError("watermark length must be positive")
        if wm_len > capacity:
            raise ValueError(
                f"Watermark length ({wm_len}) exceeds capacity ({capacity}) bits. "
                f"Use a shorter vector, a smaller block_size, a larger image, or a lower transform level."
            )
        if self.repetition:
            used_blocks = capacity
            schedule = np.arange(used_blocks, dtype=np.int64) % wm_len
        else:
            used_blocks = wm_len
            schedule = np.arange(wm_len, dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        prn = rng.integers(0, 2, size=used_blocks, dtype=np.uint8)
        return schedule, prn, used_blocks

    # ---- main API ------------------------------------------------------------

    def capacity_report(self, image_rgb: np.ndarray, watermark_length: Optional[int] = None) -> Dict[str, Any]:
        ycc = rgb_to_ycrcb_float(image_rgb)
        y = ycc[:, :, 0]
        coeffs = self.backend.forward(y)
        subband = self.backend.get_selected_subband(coeffs, self.orientation_index)
        rows, cols = subband.shape[:2]
        capacity = len(self._block_positions((rows, cols)))
        report: Dict[str, Any] = {
            "image_height": int(image_rgb.shape[0]),
            "image_width": int(image_rgb.shape[1]),
            "image_pixels": int(image_rgb.shape[0] * image_rgb.shape[1]),
            "transform": self.transform_name,
            "level": self.level,
            "orientation_index": self.orientation_index,
            "selected_subband_shape": [int(rows), int(cols)],
            "block_size": self.block_size,
            "capacity_bits_max": int(capacity),
            "capacity_bpp_max": float(capacity / (image_rgb.shape[0] * image_rgb.shape[1])),
        }
        if watermark_length is not None:
            report.update(
                {
                    "watermark_length_bits": int(watermark_length),
                    "payload_bpp": float(watermark_length / (image_rgb.shape[0] * image_rgb.shape[1])),
                    "fits_capacity": bool(watermark_length <= capacity),
                    "redundancy_factor_if_repetition": float(capacity / watermark_length) if watermark_length > 0 else None,
                }
            )
        return report

    def embed(self, image_rgb: np.ndarray, watermark_vector: np.ndarray) -> Tuple[np.ndarray, WatermarkKey]:
        """Embed watermark vector into image and return watermarked image + extraction key."""
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape HxWx3")
        bits = normalize_watermark_vector(watermark_vector)

        ycc = rgb_to_ycrcb_float(image_rgb)
        y = ycc[:, :, 0]
        coeffs = self.backend.forward(y)
        selected = self.backend.get_selected_subband(coeffs, self.orientation_index)

        if np.iscomplexobj(selected):
            selected_real = selected.real.copy()
            selected_imag = selected.imag.copy()
        else:
            selected_real = np.asarray(selected, dtype=np.float64).copy()
            selected_imag = None

        positions = self._block_positions(selected_real.shape)
        capacity = len(positions)
        schedule, prn, used_blocks = self._make_schedule_and_prn(bits.size, capacity)
        spread_bits = np.bitwise_xor(bits[schedule], prn)

        bs = self.block_size
        modified = selected_real.copy()

        for k in range(used_blocks):
            r, c = positions[k]
            block = modified[r : r + bs, c : c + bs]
            U, S, Vt = np.linalg.svd(block, full_matrices=False)
            S_mod = S.copy()
            bit = int(spread_bits[k])
            S_mod[self.svd_index] = self._qim_embed_scalar(float(S_mod[self.svd_index]), bit, self.delta)
            block_mod = (U @ np.diag(S_mod) @ Vt)
            modified[r : r + bs, c : c + bs] = block_mod

        if selected_imag is not None:
            selected_new = modified + 1j * selected_imag
        else:
            selected_new = modified

        y_w = self.backend.replace_selected_subband(
            coeffs, selected_new, self.orientation_index, original_shape=(y.shape[0], y.shape[1])
        )
        ycc_w = ycc.copy()
        ycc_w[:, :, 0] = np.clip(y_w, 0, 255)
        watermarked_rgb = ycrcb_float_to_rgb(ycc_w)

        key = WatermarkKey(
            transform=self.transform_name,
            seed=self.seed,
            watermark_length=int(bits.size),
            original_image_shape=tuple(int(x) for x in image_rgb.shape),
            delta=self.delta,
            block_size=self.block_size,
            svd_index=self.svd_index,
            level=self.level,
            orientation_index=self.orientation_index,
            highpass_shape=(int(selected_real.shape[0]), int(selected_real.shape[1])),
            capacity_bits=int(capacity),
            used_blocks=int(used_blocks),
            repetition=self.repetition,
            wavelet=self.wavelet,
        )
        return watermarked_rgb, key

    def extract(self, suspected_rgb: np.ndarray, key: WatermarkKey) -> np.ndarray:
        """Extract the original watermark vector using the watermarked/suspected image and key."""
        # Use key parameters to avoid accidental mismatch with current instance.
        extractor = VectorWatermarkSystem(
            delta=key.delta,
            block_size=key.block_size,
            seed=key.seed,
            transform=key.transform,
            level=key.level,
            orientation_index=key.orientation_index,
            svd_index=key.svd_index,
            repetition=key.repetition,
            wavelet=key.wavelet,
        )

        ycc = rgb_to_ycrcb_float(suspected_rgb)
        y = ycc[:, :, 0]
        coeffs = extractor.backend.forward(y)
        selected = extractor.backend.get_selected_subband(coeffs, key.orientation_index)
        selected_real = selected.real if np.iscomplexobj(selected) else np.asarray(selected, dtype=np.float64)

        positions = extractor._block_positions(selected_real.shape)
        if len(positions) < key.used_blocks:
            raise ValueError(
                f"Suspected image has insufficient capacity ({len(positions)} blocks) for key requiring {key.used_blocks}. "
                "Geometric attacks must be resized back to the original dimensions before extraction."
            )

        schedule, prn, used_blocks = extractor._make_schedule_and_prn(key.watermark_length, len(positions))
        # Use exactly the number stored in the key.
        schedule = schedule[: key.used_blocks]
        prn = prn[: key.used_blocks]
        used_blocks = key.used_blocks

        bs = key.block_size
        recovered_spread = np.zeros(used_blocks, dtype=np.uint8)
        for k in range(used_blocks):
            r, c = positions[k]
            block = selected_real[r : r + bs, c : c + bs]
            _, S, _ = np.linalg.svd(block, full_matrices=False)
            recovered_spread[k] = extractor._qim_extract_scalar(float(S[key.svd_index]), key.delta)

        recovered_bits_per_block = np.bitwise_xor(recovered_spread, prn)

        if key.repetition:
            # Majority vote for each original bit.
            votes = np.zeros((key.watermark_length, 2), dtype=np.int64)
            for bit_index, value in zip(schedule, recovered_bits_per_block):
                votes[int(bit_index), int(value)] += 1
            recovered = (votes[:, 1] >= votes[:, 0]).astype(np.uint8)
        else:
            recovered = recovered_bits_per_block[: key.watermark_length].astype(np.uint8)
        return recovered

    def verify(
        self,
        original_watermark: np.ndarray,
        extracted_watermark: np.ndarray,
        ncc_threshold: float = 0.90,
        ber_threshold: float = 0.10,
    ) -> Dict[str, Any]:
        metrics = watermark_similarity_metrics(original_watermark, extracted_watermark)
        accepted = (metrics["NCC"] >= ncc_threshold) and (metrics["BER"] <= ber_threshold)
        metrics.update(
            {
                "NCC_threshold": float(ncc_threshold),
                "BER_threshold": float(ber_threshold),
                "Watermark_Accepted": bool(accepted),
            }
        )
        return metrics


# -----------------------------------------------------------------------------
# Robustness attacks
# -----------------------------------------------------------------------------


def attack_jpeg(image_rgb: np.ndarray, quality: int = 70) -> np.ndarray:
    bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def attack_gaussian_noise(image_rgb: np.ndarray, sigma: float = 5.0, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image_rgb.astype(np.float64) + rng.normal(0, sigma, size=image_rgb.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def attack_salt_pepper(image_rgb: np.ndarray, amount: float = 0.01, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image_rgb.copy()
    h, w = out.shape[:2]
    n = int(amount * h * w)
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    vals = rng.choice([0, 255], size=n)
    out[ys, xs, :] = vals[:, None]
    return out


def attack_speckle_noise(image_rgb: np.ndarray, sigma: float = 0.05, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = image_rgb.astype(np.float64)
    noisy = img + img * rng.normal(0, sigma, size=img.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def attack_median_filter(image_rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(image_rgb, ksize)


def attack_gaussian_blur(image_rgb: np.ndarray, ksize: int = 3, sigma: float = 0.8) -> np.ndarray:
    return cv2.GaussianBlur(image_rgb, (ksize, ksize), sigmaX=sigma)


def attack_average_filter(image_rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.blur(image_rgb, (ksize, ksize))


def attack_sharpen(image_rgb: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
    out = cv2.filter2D(image_rgb, -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def attack_hist_equalization(image_rgb: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def attack_contrast_brightness(image_rgb: np.ndarray, alpha: float = 1.15, beta: float = 5.0) -> np.ndarray:
    out = alpha * image_rgb.astype(np.float64) + beta
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def attack_resize_restore(image_rgb: np.ndarray, scale: float = 0.75) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    small = cv2.resize(image_rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return restored


def attack_rotate_restore(image_rgb: np.ndarray, angle: float = 5.0) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # In a real blind attack, alignment may be unknown. Here we keep dimensions fixed.
    return rotated


def attack_center_crop_resize(image_rgb: np.ndarray, crop_ratio: float = 0.90) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    crop = image_rgb[y0 : y0 + ch, x0 : x0 + cw]
    restored = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    return restored


def attack_random_occlusion(image_rgb: np.ndarray, ratio: float = 0.10, seed: int = 123) -> np.ndarray:
    """Local tampering/occlusion: black rectangle covering ratio of image area."""
    rng = np.random.default_rng(seed)
    out = image_rgb.copy()
    h, w = out.shape[:2]
    area = h * w * ratio
    rect_h = int(math.sqrt(area))
    rect_w = int(area / max(rect_h, 1))
    rect_h = max(1, min(rect_h, h))
    rect_w = max(1, min(rect_w, w))
    y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
    x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
    out[y0 : y0 + rect_h, x0 : x0 + rect_w, :] = 0
    return out


def default_attack_suite() -> List[Tuple[str, Any]]:
    """Return a list of named attack callables."""
    return [
        ("No attack", lambda img: img.copy()),
        ("JPEG Q=90", lambda img: attack_jpeg(img, 90)),
        ("JPEG Q=70", lambda img: attack_jpeg(img, 70)),
        ("JPEG Q=50", lambda img: attack_jpeg(img, 50)),
        ("JPEG Q=30", lambda img: attack_jpeg(img, 30)),
        ("Gaussian noise sigma=3", lambda img: attack_gaussian_noise(img, 3.0, seed=1)),
        ("Gaussian noise sigma=5", lambda img: attack_gaussian_noise(img, 5.0, seed=2)),
        ("Salt-pepper amount=0.005", lambda img: attack_salt_pepper(img, 0.005, seed=3)),
        ("Salt-pepper amount=0.01", lambda img: attack_salt_pepper(img, 0.01, seed=4)),
        ("Speckle sigma=0.03", lambda img: attack_speckle_noise(img, 0.03, seed=5)),
        ("Median filter 3x3", lambda img: attack_median_filter(img, 3)),
        ("Gaussian blur 3x3", lambda img: attack_gaussian_blur(img, 3, 0.8)),
        ("Average filter 3x3", lambda img: attack_average_filter(img, 3)),
        ("Sharpening", lambda img: attack_sharpen(img)),
        ("Histogram equalization", lambda img: attack_hist_equalization(img)),
        ("Contrast alpha=1.15 beta=5", lambda img: attack_contrast_brightness(img, 1.15, 5.0)),
        ("Resize 75% restore", lambda img: attack_resize_restore(img, 0.75)),
        ("Resize 50% restore", lambda img: attack_resize_restore(img, 0.50)),
        ("Rotation 2 deg", lambda img: attack_rotate_restore(img, 2.0)),
        ("Rotation 5 deg", lambda img: attack_rotate_restore(img, 5.0)),
        ("Center crop 90% restore", lambda img: attack_center_crop_resize(img, 0.90)),
        ("Center crop 80% restore", lambda img: attack_center_crop_resize(img, 0.80)),
        ("Random occlusion 5%", lambda img: attack_random_occlusion(img, 0.05, seed=6)),
        ("Random occlusion 10%", lambda img: attack_random_occlusion(img, 0.10, seed=7)),
    ]


def evaluate_robustness(
    system: VectorWatermarkSystem,
    watermarked_rgb: np.ndarray,
    original_watermark: np.ndarray,
    key: WatermarkKey,
    ncc_threshold: float = 0.90,
    ber_threshold: float = 0.10,
    save_attacked_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Run default attacks and return a DataFrame of extraction/verification results."""
    rows: List[Dict[str, Any]] = []
    attack_dir: Optional[Path] = None
    if save_attacked_dir is not None:
        attack_dir = ensure_dir(save_attacked_dir)

    for name, fn in default_attack_suite():
        try:
            attacked = fn(watermarked_rgb)
            extracted = system.extract(attacked, key)
            metrics = system.verify(original_watermark, extracted, ncc_threshold, ber_threshold)
            row = {"Attack": name, **metrics}
            rows.append(row)
            if attack_dir is not None:
                safe_name = name.replace(" ", "_").replace("=", "").replace("%", "pct").replace("/", "_")
                save_image_rgb(attack_dir / f"{safe_name}.png", attacked)
        except Exception as exc:
            rows.append({"Attack": name, "Error": str(exc), "Watermark_Accepted": False})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Key sensitivity / wrong-key test
# -----------------------------------------------------------------------------


def evaluate_key_sensitivity(
    image_rgb: np.ndarray,
    original_watermark: np.ndarray,
    key: WatermarkKey,
    wrong_seeds: Sequence[int] = (11, 22, 33, 44, 55),
    ncc_threshold: float = 0.90,
    ber_threshold: float = 0.10,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    correct_system = VectorWatermarkSystem(
        delta=key.delta,
        block_size=key.block_size,
        seed=key.seed,
        transform=key.transform,
        level=key.level,
        orientation_index=key.orientation_index,
        svd_index=key.svd_index,
        repetition=key.repetition,
        wavelet=key.wavelet,
    )
    extracted_correct = correct_system.extract(image_rgb, key)
    rows.append({"Condition": "Correct key", "Seed": key.seed, **correct_system.verify(original_watermark, extracted_correct, ncc_threshold, ber_threshold)})

    for seed in wrong_seeds:
        wrong_key = WatermarkKey(
            transform=key.transform,
            seed=int(seed),
            watermark_length=key.watermark_length,
            original_image_shape=key.original_image_shape,
            delta=key.delta,
            block_size=key.block_size,
            svd_index=key.svd_index,
            level=key.level,
            orientation_index=key.orientation_index,
            highpass_shape=key.highpass_shape,
            capacity_bits=key.capacity_bits,
            used_blocks=key.used_blocks,
            repetition=key.repetition,
            wavelet=key.wavelet,
        )
        wrong_system = VectorWatermarkSystem(
            delta=wrong_key.delta,
            block_size=wrong_key.block_size,
            seed=wrong_key.seed,
            transform=wrong_key.transform,
            level=wrong_key.level,
            orientation_index=wrong_key.orientation_index,
            svd_index=wrong_key.svd_index,
            repetition=wrong_key.repetition,
            wavelet=wrong_key.wavelet,
        )
        extracted_wrong = wrong_system.extract(image_rgb, wrong_key)
        rows.append({"Condition": "Wrong key", "Seed": int(seed), **wrong_system.verify(original_watermark, extracted_wrong, ncc_threshold, ber_threshold)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI / complete experiment runner
# -----------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)

    face_rgb = load_image_rgb(args.face)

    if args.watermark:
        watermark = load_watermark_vector(args.watermark)
    else:
        rng = np.random.default_rng(args.seed)
        watermark = rng.integers(0, 2, size=int(args.demo_wm_len), dtype=np.uint8)

    watermark = enforce_minimum_watermark_length(watermark, min_bits=args.min_wm_bits, mode="repeat")
    save_watermark_vector(out_dir / "watermark_used.npy", watermark)
    save_watermark_vector(out_dir / "watermark_used.txt", watermark)

    system = VectorWatermarkSystem(
        delta=args.delta,
        block_size=args.block_size,
        seed=args.seed,
        transform=args.transform,
        level=args.level,
        orientation_index=args.orientation_index,
        svd_index=args.svd_index,
        repetition=not args.no_repetition,
        wavelet=args.wavelet,
    )

    # 1) capacity before embedding
    capacity = system.capacity_report(face_rgb, watermark_length=len(watermark))
    with open(out_dir / "capacity_report.json", "w", encoding="utf-8") as f:
        json.dump(capacity, f, indent=2)

    # 2) embedding
    watermarked_rgb, key = system.embed(face_rgb, watermark)
    save_image_rgb(out_dir / "watermarked.png", watermarked_rgb)
    key.to_json(out_dir / "watermark_key.json")

    # 3) clean extraction + verification
    extracted_clean = system.extract(watermarked_rgb, key)
    save_watermark_vector(out_dir / "extracted_clean.npy", extracted_clean)
    save_watermark_vector(out_dir / "extracted_clean.txt", extracted_clean)

    imperceptibility = image_imperceptibility_metrics(face_rgb, watermarked_rgb)
    watermark_metrics = system.verify(watermark, extracted_clean, args.ncc_threshold, args.ber_threshold)
    clean_report = {"Imperceptibility": imperceptibility, "Watermark_Verification": watermark_metrics}
    with open(out_dir / "clean_metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean_report, f, indent=2)

    # 4) robustness
    robustness_df = evaluate_robustness(
        system,
        watermarked_rgb,
        watermark,
        key,
        ncc_threshold=args.ncc_threshold,
        ber_threshold=args.ber_threshold,
        save_attacked_dir=(out_dir / "attacked_images") if args.save_attacked_images else None,
    )
    robustness_df.to_csv(out_dir / "robustness_results.csv", index=False)
    robustness_df.to_excel(out_dir / "robustness_results.xlsx", index=False)

    # 5) key sensitivity
    key_df = evaluate_key_sensitivity(
        watermarked_rgb,
        watermark,
        key,
        wrong_seeds=[args.seed + 1, args.seed + 7, args.seed + 19, args.seed + 101, args.seed + 1009],
        ncc_threshold=args.ncc_threshold,
        ber_threshold=args.ber_threshold,
    )
    key_df.to_csv(out_dir / "key_sensitivity_results.csv", index=False)
    key_df.to_excel(out_dir / "key_sensitivity_results.xlsx", index=False)

    # Console summary
    print("\n=== Watermark experiment completed ===")
    print(f"Output directory: {out_dir.resolve()}")
    print("\n--- Capacity ---")
    print(json.dumps(capacity, indent=2))
    print("\n--- Clean imperceptibility ---")
    print(json.dumps(imperceptibility, indent=2))
    print("\n--- Clean watermark verification ---")
    print(json.dumps(watermark_metrics, indent=2))
    print("\n--- Robustness table saved to robustness_results.csv/xlsx ---")
    print(robustness_df[[c for c in ["Attack", "NCC", "BER_percent", "Extraction_Accuracy_percent", "Watermark_Accepted", "Error"] if c in robustness_df.columns]].to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vector watermark embedding/extraction/evaluation for biometric face watermarking.")
    p.add_argument("--face", required=True, help="Path to face/host image.")
    p.add_argument("--watermark", default=None, help="Path to binary watermark vector (.npy, .txt, .csv, .json). If omitted, demo vector is generated.")
    p.add_argument("--demo_wm_len", type=int, default=128, help="Random watermark length when --watermark is omitted.")
    p.add_argument("--min_wm_bits", type=int, default=MIN_WATERMARK_BITS, help="Minimum watermark length. Shorter vectors are repeated to this length.")
    p.add_argument("--out_dir", default="watermark_results", help="Output folder.")
    p.add_argument("--transform", choices=["dtcwt", "dwt"], default="dtcwt", help="Use dtcwt for the paper method; dwt is only a fallback test mode.")
    p.add_argument("--delta", type=float, default=4.0, help="QIM/SVD embedding strength. Larger = more robust, less imperceptible.")
    p.add_argument("--block_size", type=int, default=8, help="SVD block size in selected subband.")
    p.add_argument("--seed", type=int, default=2026, help="Secret seed for pseudo-random spreading.")
    p.add_argument("--level", type=int, default=2, help="Transform decomposition level. Paper uses 2.")
    p.add_argument("--orientation_index", type=int, default=2, help="DT-CWT orientation 0..5; DWT fallback: 0=cH,1=cV,2=cD.")
    p.add_argument("--svd_index", type=int, default=0, help="Singular value index to embed into. 0 is most robust.")
    p.add_argument("--no_repetition", action="store_true", help="Disable repetition coding over all available blocks.")
    p.add_argument("--wavelet", default="haar", help="Wavelet for DWT fallback only.")
    p.add_argument("--ncc_threshold", type=float, default=0.90, help="NCC threshold for watermark acceptance.")
    p.add_argument("--ber_threshold", type=float, default=0.10, help="BER threshold for watermark acceptance.")
    p.add_argument("--save_attacked_images", action="store_true", help="Save attacked images used in robustness evaluation.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    run_experiment(parser.parse_args())
