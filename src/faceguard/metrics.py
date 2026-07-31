from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .io import ensure_rgb_uint8, to_binary_vector


def image_quality(reference: np.ndarray, query: np.ndarray) -> dict[str, float]:
    reference = ensure_rgb_uint8(reference)
    query = ensure_rgb_uint8(query)
    if reference.shape != query.shape:
        query = cv2.resize(query, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    ref = reference.astype(np.float64)
    qry = query.astype(np.float64)
    mse = float(np.mean((ref - qry) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0**2) / mse))
    ssim = float(structural_similarity(reference, query, channel_axis=-1, data_range=255))
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


def bipolar_ncc(original: np.ndarray, extracted: np.ndarray) -> float:
    a = to_binary_vector(original)
    b = to_binary_vector(extracted)
    n = min(a.size, b.size)
    if n == 0:
        raise ValueError("Cannot compare empty watermark vectors.")
    a = 2.0 * a[:n].astype(np.float64) - 1.0
    b = 2.0 * b[:n].astype(np.float64) - 1.0
    return float(np.mean(a * b))


def watermark_metrics(
    original: np.ndarray,
    extracted: np.ndarray,
    ncc_threshold: float = 0.90,
    ber_threshold: float = 0.05,
) -> dict[str, float | int | bool]:
    a = to_binary_vector(original)
    b = to_binary_vector(extracted)
    n = min(a.size, b.size)
    if n == 0:
        raise ValueError("Cannot compare empty watermark vectors.")
    errors = int(np.sum(a[:n] != b[:n]))
    ber = errors / n
    ncc = bipolar_ncc(a[:n], b[:n])
    return {
        "NCC": ncc,
        "BER": ber,
        "BER_percent": 100.0 * ber,
        "Extraction_Accuracy": 1.0 - ber,
        "Extraction_Accuracy_percent": 100.0 * (1.0 - ber),
        "Bit_Errors": errors,
        "Effective_Bits": n,
        "Watermark_Accepted": bool(ncc >= ncc_threshold and ber <= ber_threshold),
    }
