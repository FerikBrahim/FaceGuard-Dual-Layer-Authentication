#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import gdown


def normalize(source: str) -> str:
    source = source.strip()
    if "drive.google.com" in source or "drive.usercontent.google.com" in source:
        return source
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", source):
        return f"https://drive.google.com/file/d/{source}/view?usp=sharing"
    raise ValueError("Supply a Google Drive sharing URL or raw file ID.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download face_verification_model.pth from Google Drive.")
    parser.add_argument("source")
    parser.add_argument("--output", default="models/face_verification_model.pth")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(url=normalize(args.source), output=str(destination), quiet=False)
    if result is None or not destination.exists():
        raise RuntimeError("Checkpoint download failed. Set Drive access to 'Anyone with the link'.")
    print(f"Saved checkpoint to {destination}")


if __name__ == "__main__":
    main()
