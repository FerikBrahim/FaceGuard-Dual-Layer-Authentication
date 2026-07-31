import numpy as np

from faceguard.watermark.compcode import CompCodeGenerator


def test_compcode_is_deterministic_and_128_bits():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    for row in range(256):
        image[row, :, :] = row
    generator = CompCodeGenerator(output_bits=128, seed=2026)
    first = generator.generate(image)
    second = generator.generate(image)
    assert first.shape == (128,)
    assert np.array_equal(first, second)
    assert set(np.unique(first)).issubset({0, 1})
