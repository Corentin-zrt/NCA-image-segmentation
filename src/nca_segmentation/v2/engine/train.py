from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from ..metrics.segmentation import mean_iou


@dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    ignore_index: int = 255
    checkpoint_dir: Path = Path("artifacts/runs/v2")


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, config: TrainConfig) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        image = _tensor(batch["image"], device)
        target = _tensor(batch["semantic"], device).long()
        output = model(image)
        semantic_loss = nn.functional.cross_entropy(output["semantic_logits"], target, ignore_index=config.ignore_index)
        object_target = (target > 0) & (target != config.ignore_index)
        object_loss = nn.functional.binary_cross_entropy_with_logits(output["objectness_logits"].squeeze(1), object_target.float())
        loss = semantic_loss + 0.25 * object_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(1, batches)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, config: TrainConfig) -> dict[str, float]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            image = _tensor(batch["image"], device)
            target = _tensor(batch["semantic"], device).long()
            predictions.append(model(image)["semantic_logits"].argmax(dim=1).cpu())
            targets.append(target.cpu())
    return mean_iou(torch.cat(predictions), torch.cat(targets), model.config.num_classes, config.ignore_index)


def fit(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, config: TrainConfig, device: torch.device | str = "cpu") -> list[dict[str, float]]:
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, config)
        metrics = evaluate(model, validation_loader, device, config)
        metrics["loss"] = loss
        metrics["epoch"] = float(epoch)
        save_checkpoint(config.checkpoint_dir / "last.pt", model, optimizer, epoch, metrics)
        history.append(metrics)
    return history


def _tensor(value: object, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)
