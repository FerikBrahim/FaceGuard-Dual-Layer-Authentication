from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AuthenticationOutcome(str, Enum):
    GENUINE = "Genuine"
    WATERMARK_TAMPERING = "Watermark tampering"
    IDENTITY_SPOOFING = "Identity spoofing or face-swap detected"
    REJECTED = "Rejected"
    SIAMESE_ONLY_ACCEPTED = "Siamese-only accepted"
    SIAMESE_ONLY_REJECTED = "Siamese-only rejected"


@dataclass(frozen=True)
class FusionResult:
    A_s: int
    A_w: int | None
    A_f: int | None
    outcome: AuthenticationOutcome


def fuse_decisions(siamese_accepted: bool, watermark_accepted: bool | None) -> FusionResult:
    A_s = int(siamese_accepted)
    if watermark_accepted is None:
        return FusionResult(
            A_s=A_s,
            A_w=None,
            A_f=None,
            outcome=AuthenticationOutcome.SIAMESE_ONLY_ACCEPTED if siamese_accepted else AuthenticationOutcome.SIAMESE_ONLY_REJECTED,
        )
    A_w = int(watermark_accepted)
    A_f = int(siamese_accepted and watermark_accepted)
    if A_s == 1 and A_w == 1:
        outcome = AuthenticationOutcome.GENUINE
    elif A_s == 1 and A_w == 0:
        outcome = AuthenticationOutcome.WATERMARK_TAMPERING
    elif A_s == 0 and A_w == 1:
        outcome = AuthenticationOutcome.IDENTITY_SPOOFING
    else:
        outcome = AuthenticationOutcome.REJECTED
    return FusionResult(A_s=A_s, A_w=A_w, A_f=A_f, outcome=outcome)
