from faceguard.fusion.authentication import AuthenticationOutcome, fuse_decisions


def test_all_fusion_outcomes():
    assert fuse_decisions(True, True).outcome == AuthenticationOutcome.GENUINE
    assert fuse_decisions(True, False).outcome == AuthenticationOutcome.WATERMARK_TAMPERING
    assert fuse_decisions(False, True).outcome == AuthenticationOutcome.IDENTITY_SPOOFING
    assert fuse_decisions(False, False).outcome == AuthenticationOutcome.REJECTED


def test_siamese_only_outcome():
    result = fuse_decisions(True, None)
    assert result.A_w is None
    assert result.A_f is None
    assert result.outcome == AuthenticationOutcome.SIAMESE_ONLY_ACCEPTED
