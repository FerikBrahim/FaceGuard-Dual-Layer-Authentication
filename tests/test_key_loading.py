import json

from faceguard.watermark.dtcwt_svd import PairwiseSVDKey


def test_key_loader_ignores_descriptive_metadata(tmp_path):
    payload = {
        "seed": 2026,
        "watermark_length": 128,
        "block_size": 4,
        "level": 2,
        "orientation_index": 2,
        "margin": 36.0,
        "repetitions": 5,
        "svd_index": 0,
        "original_shape": [512, 512, 3],
        "working_shape": [512, 512, 3],
        "pairs": [],
        "spread_mask": [0] * 128,
        "test_only": True,
        "note": "ignored metadata",
    }
    path = tmp_path / "key.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    key = PairwiseSVDKey.from_json(path)
    assert key.watermark_length == 128
    assert not hasattr(key, "test_only")
