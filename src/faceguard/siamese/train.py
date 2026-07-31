from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .loss import CosineTripletLoss
from .model import FaceTripletModel


def train_triplet_model(
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    output_checkpoint: str | Path,
    epochs: int = 15,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    margin: float = 0.8,
    embedding_dim: int = 512,
    device: torch.device | None = None,
) -> FaceTripletModel:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceTripletModel(embedding_dim=embedding_dim, pretrained="vggface2").to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = CosineTripletLoss(margin)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        batches = 0
        for anchor, positive, negative, *_ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            embeddings = model(anchor, positive, negative)
            loss = criterion(*embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        epoch_loss = total / max(batches, 1)
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss})
        print(f"Epoch {epoch + 1:02d}/{epochs}: loss={epoch_loss:.6f}")
    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_history": history,
            "embedding_dim": embedding_dim,
        },
        output_checkpoint,
    )
    output_checkpoint.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model
