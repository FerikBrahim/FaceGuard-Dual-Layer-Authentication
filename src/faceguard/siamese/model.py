from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from faceguard.io import ensure_rgb_uint8

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
except ImportError as exc:  # pragma: no cover
    InceptionResnetV1 = None
    MTCNN = None
    _FACENET_IMPORT_ERROR = exc
else:
    _FACENET_IMPORT_ERROR = None


class FaceEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, pretrained: str | None = "vggface2") -> None:
        super().__init__()
        if InceptionResnetV1 is None:
            raise ImportError("Install facenet-pytorch to use the Siamese module.") from _FACENET_IMPORT_ERROR
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if hasattr(self.backbone, "last_linear"):
            for parameter in self.backbone.last_linear.parameters():
                parameter.requires_grad = True
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone(inputs)), p=2, dim=1)


class FaceTripletModel(nn.Module):
    def __init__(self, embedding_dim: int = 512, pretrained: str | None = "vggface2") -> None:
        super().__init__()
        self.encoder = FaceEncoder(embedding_dim=embedding_dim, pretrained=pretrained)

    def forward_one(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        return self.forward_one(anchor), self.forward_one(positive), self.forward_one(negative)


class FacePreprocessor:
    def __init__(self, face_size: int, device: torch.device, min_probability: float = 0.90) -> None:
        if MTCNN is None:
            raise ImportError("Install facenet-pytorch to use MTCNN.") from _FACENET_IMPORT_ERROR
        self.min_probability = float(min_probability)
        self.mtcnn = MTCNN(
            image_size=face_size,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True,
            device=device,
        )
        self.fallback = transforms.Compose(
            [
                transforms.Resize((face_size, face_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def process(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, float, bool]:
        image = Image.fromarray(ensure_rgb_uint8(image_rgb))
        face, probability = self.mtcnn(image, return_prob=True)
        if face is not None and probability is not None and float(probability) >= self.min_probability:
            return face, float(probability), True
        return self.fallback(image), float(probability or 0.0), False


@dataclass
class SiameseResult:
    similarity: float
    distance: float
    threshold: float
    accepted: bool
    reference_face_probability: float
    query_face_probability: float
    reference_face_detected: bool
    query_face_detected: bool
    runtime_ms: float


class SiameseVerifier:
    def __init__(
        self,
        checkpoint_path: str | Path,
        metrics_json: str | Path | None = None,
        embedding_dim: int = 512,
        face_size: int = 160,
        threshold: float | None = None,
        min_face_probability: float = 0.90,
        trusted_checkpoint: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceTripletModel(embedding_dim=embedding_dim, pretrained=None).to(self.device)
        self.model.eval()
        self.preprocessor = FacePreprocessor(face_size, self.device, min_face_probability)
        checkpoint = self._load_checkpoint(checkpoint_path, trusted_checkpoint)
        state = checkpoint
        if isinstance(checkpoint, dict):
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint.get("model", checkpoint)))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print("Warning: missing checkpoint keys:", missing)
        if unexpected:
            print("Warning: unexpected checkpoint keys:", unexpected)
        self.threshold = self._resolve_threshold(threshold, metrics_json, checkpoint)

    def _load_checkpoint(self, path: str | Path, trusted: bool) -> Any:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            return torch.load(path, map_location=self.device, weights_only=not trusted)
        except TypeError:
            if not trusted:
                raise RuntimeError("This PyTorch version requires trusted_checkpoint=True for legacy pickle checkpoints.")
            return torch.load(path, map_location=self.device)
        except Exception as exc:
            if not trusted:
                raise RuntimeError(
                    "Restricted checkpoint loading failed. Set trusted_checkpoint=True only for a checkpoint you trust."
                ) from exc
            raise

    @staticmethod
    def _resolve_threshold(explicit: float | None, metrics_json: str | Path | None, checkpoint: Any) -> float:
        if explicit is not None:
            return float(explicit)
        if metrics_json is not None and Path(metrics_json).exists():
            metrics = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
            for key in ("optimal_threshold", "eer_threshold"):
                if key in metrics:
                    return float(metrics[key])
        if isinstance(checkpoint, dict):
            metrics = checkpoint.get("final_metrics", {})
            for key in ("optimal_threshold", "eer_threshold"):
                if key in metrics:
                    return float(metrics[key])
        raise ValueError("Provide a validation-derived Siamese threshold or metrics JSON.")

    def embed(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, float, bool]:
        tensor, probability, detected = self.preprocessor.process(image_rgb)
        with torch.no_grad():
            embedding = self.model.forward_one(tensor.unsqueeze(0).to(self.device))
        return embedding, probability, detected

    def verify(self, reference_rgb: np.ndarray, query_rgb: np.ndarray) -> SiameseResult:
        started = time.perf_counter()
        reference, ref_probability, ref_detected = self.embed(reference_rgb)
        query, query_probability, query_detected = self.embed(query_rgb)
        similarity = float(F.cosine_similarity(reference, query).item())
        return SiameseResult(
            similarity=similarity,
            distance=1.0 - similarity,
            threshold=self.threshold,
            accepted=bool(similarity >= self.threshold),
            reference_face_probability=ref_probability,
            query_face_probability=query_probability,
            reference_face_detected=ref_detected,
            query_face_detected=query_detected,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
        )
