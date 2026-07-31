import numpy as np

from faceguard.metrics import bipolar_ncc, image_quality, watermark_metrics


def test_watermark_metrics_perfect_match():
    bits = np.tile([0, 1], 64).astype(np.uint8)
    metrics = watermark_metrics(bits, bits)
    assert metrics["NCC"] == 1.0
    assert metrics["BER"] == 0.0
    assert metrics["Watermark_Accepted"] is True


def test_watermark_metrics_known_errors():
    original = np.zeros(128, dtype=np.uint8)
    extracted = original.copy()
    extracted[:4] = 1
    metrics = watermark_metrics(original, extracted, ncc_threshold=0.90, ber_threshold=0.05)
    assert metrics["Bit_Errors"] == 4
    assert metrics["BER_percent"] == 3.125
    assert metrics["NCC"] == 0.9375
    assert metrics["Watermark_Accepted"] is True


def test_image_quality_self_comparison():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    metrics = image_quality(image, image)
    assert metrics["MSE"] == 0.0
    assert np.isinf(metrics["PSNR"])
    assert metrics["SSIM"] == 1.0
