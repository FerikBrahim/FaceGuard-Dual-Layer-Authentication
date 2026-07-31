from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from faceguard.siamese.evaluate import threshold_metrics

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError as exc:  # pragma: no cover
    InceptionResnetV1 = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class VerificationPairDataset(Dataset):
    def __init__(self, pairs: Iterable[tuple[str | Path, str | Path, int]], face_size: int = 160) -> None:
        self.pairs = [(Path(first), Path(second), int(label)) for first, second, label in pairs]
        self.transform = transforms.Compose(
            [
                transforms.Resize((face_size, face_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        first, second, label = self.pairs[index]
        return (
            self.transform(Image.open(first).convert("RGB")),
            self.transform(Image.open(second).convert("RGB")),
            label,
        )


def evaluate_vggface2_zero_shot(
    dataloader: DataLoader, device: torch.device | None = None
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if InceptionResnetV1 is None:
        raise ImportError("Install facenet-pytorch for zero-shot evaluation") from _IMPORT_ERROR
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for first, second, batch_labels in dataloader:
            first_embedding = F.normalize(model(first.to(device)), p=2, dim=1)
            second_embedding = F.normalize(model(second.to(device)), p=2, dim=1)
            scores.extend(F.cosine_similarity(first_embedding, second_embedding).cpu().tolist())
            labels.extend(torch.as_tensor(batch_labels).cpu().int().tolist())
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int32)
    return threshold_metrics(label_array, score_array), score_array, label_array
