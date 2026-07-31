from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def images_by_identity(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    mapping: dict[str, list[Path]] = {}
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = sorted(path for path in identity_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            mapping[identity_dir.name] = images
    return mapping


def identity_disjoint_split(identities: list[str], validation_fraction: float = 0.2, seed: int = 2026):
    values = identities.copy()
    random.Random(seed).shuffle(values)
    validation_count = max(1, int(round(len(values) * validation_fraction)))
    validation = set(values[:validation_count])
    training = set(values[validation_count:])
    return training, validation


class FaceTripletDataset(Dataset):
    """Triplets from identity-named subfolders.

    `anchor_root/<identity>` and `positive_root/<identity>` must describe the
    same person. Negatives are sampled from a different identity in
    `negative_root`.
    """

    def __init__(
        self,
        anchor_root: str | Path,
        positive_root: str | Path,
        negative_root: str | Path,
        identities: set[str] | None = None,
        triplets_per_identity: int = 20,
        face_size: int = 160,
        seed: int = 2026,
    ) -> None:
        self.anchor = images_by_identity(anchor_root)
        self.positive = images_by_identity(positive_root)
        self.negative = images_by_identity(negative_root)
        common = sorted(set(self.anchor) & set(self.positive))
        if identities is not None:
            common = [identity for identity in common if identity in identities]
        if len(common) < 2:
            raise ValueError("At least two identities with anchor and positive images are required.")
        rng = random.Random(seed)
        negative_identities = sorted(self.negative)
        self.triplets: list[tuple[Path, Path, Path, str, str]] = []
        for identity in common:
            other = [candidate for candidate in negative_identities if candidate != identity]
            if not other:
                continue
            for _ in range(triplets_per_identity):
                negative_identity = rng.choice(other)
                self.triplets.append(
                    (
                        rng.choice(self.anchor[identity]),
                        rng.choice(self.positive[identity]),
                        rng.choice(self.negative[negative_identity]),
                        identity,
                        negative_identity,
                    )
                )
        self.transform = transforms.Compose(
            [
                transforms.Resize((face_size, face_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.triplets)

    def _load(self, path: Path) -> torch.Tensor:
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, index: int):
        anchor, positive, negative, identity, negative_identity = self.triplets[index]
        return self._load(anchor), self._load(positive), self._load(negative), identity, negative_identity
