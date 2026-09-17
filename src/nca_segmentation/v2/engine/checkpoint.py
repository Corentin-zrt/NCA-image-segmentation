from __future__ import annotations

from pathlib import Path

import torch

from nca_segmentation.v2.models.nca import StreetSceneNCA, StreetSceneNCAConfig


def save_checkpoint(
    path: Path,
    model: StreetSceneNCA,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "config": model.export_config(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[StreetSceneNCA, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = StreetSceneNCA(StreetSceneNCAConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return model, payload
