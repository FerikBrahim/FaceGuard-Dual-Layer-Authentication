from faceguard.siamese.deepfake import DeepfakeTrial, anti_spoofing_metrics


def test_anti_spoofing_metrics():
    trials = [
        DeepfakeTrial("a", "genuine", "a1.jpg", 0.9, 0.1, False),
        DeepfakeTrial("b", "genuine", "b1.jpg", 0.8, 0.2, False),
        DeepfakeTrial("a", "swap", "a_fake.jpg", 0.2, 0.8, True),
        DeepfakeTrial("b", "swap", "b_fake.jpg", 0.7, 0.3, True),
    ]
    metrics = anti_spoofing_metrics(trials, similarity_threshold=0.5)
    assert metrics["BPCER"] == 0.0
    assert metrics["APCER"] == 0.5
    assert metrics["per_method"]["swap"]["trials"] == 2
