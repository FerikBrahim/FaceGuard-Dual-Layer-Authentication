#!/usr/bin/env python3
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from faceguard.siamese.data import FaceTripletDataset, identity_disjoint_split, images_by_identity
from faceguard.siamese.train import train_triplet_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FaceGuard Siamese triplet model.")
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--positive-root", required=True)
    parser.add_argument("--negative-root", required=True)
    parser.add_argument("--checkpoint", default="models/face_verification_model.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--triplets-per-identity", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    identities = sorted(set(images_by_identity(args.anchor_root)) & set(images_by_identity(args.positive_root)))
    train_ids, validation_ids = identity_disjoint_split(identities, seed=args.seed)
    train_data = FaceTripletDataset(
        args.anchor_root,
        args.positive_root,
        args.negative_root,
        identities=train_ids,
        triplets_per_identity=args.triplets_per_identity,
        seed=args.seed,
    )
    validation_data = FaceTripletDataset(
        args.anchor_root,
        args.positive_root,
        args.negative_root,
        identities=validation_ids,
        triplets_per_identity=args.triplets_per_identity,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
    train_triplet_model(train_loader, validation_loader, args.checkpoint, epochs=args.epochs)


if __name__ == "__main__":
    main()
