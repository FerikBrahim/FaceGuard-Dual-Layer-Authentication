import numpy as np

from faceguard.attacks.facial_manipulation import generate_facial_attack_suite


def test_attack_suite_preserves_dimensions():
    target = np.full((512, 512, 3), 128, dtype=np.uint8)
    donor = np.full((512, 512, 3), 160, dtype=np.uint8)
    suite = generate_facial_attack_suite(target, donor)
    assert len(suite) == 7
    assert all(sample.image.shape == (512, 512, 3) for sample in suite)
