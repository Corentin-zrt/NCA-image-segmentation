from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from nca_segmentation.v2.data import CityscapesInstances
from nca_segmentation.v2.engine import TrainConfig, fit, get_device
from nca_segmentation.v2.models import StreetSceneNCA, StreetSceneNCAConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CPU-first Cityscapes NCA")
    parser.add_argument("--root", type=Path, required=True, help="Cityscapes root containing leftImg8bit and gtFine")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--classes", type=int, default=19)
    args = parser.parse_args()

    train = CityscapesInstances(args.root, "train", (args.height, args.width))
    validation = CityscapesInstances(args.root, "val", (args.height, args.width))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = StreetSceneNCA(StreetSceneNCAConfig(num_classes=args.classes))
    history = fit(model, train_loader, validation_loader, TrainConfig(epochs=args.epochs), get_device())
    print(history[-1])


if __name__ == "__main__":
    main()
