"""Checkpoint persistence for models and complete resolved experiments."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ExperimentConfig, ModelConfig
from .model import build_model


def _legacy_model_config(value: dict) -> ModelConfig:
    return ModelConfig.from_dict(value)


def checkpoint_model_config(blob: dict) -> ModelConfig:
    if "experiment_config" in blob:
        return ExperimentConfig.from_dict(blob["experiment_config"]).model
    value = blob["config"]
    if "model" in value: return ExperimentConfig.from_dict(value).model
    return _legacy_model_config(value)


def save_checkpoint(path, model, optimizer, scheduler, step: int, tokens_seen: int,
                    config: ExperimentConfig) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model_to_save = getattr(model, "_orig_mod", model)
    resolved = config.to_dict()
    torch.save({
        "model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "step": step, "tokens_seen": tokens_seen,
        "experiment_config": resolved,
    }, path / "checkpoint.pt")
    (path / "config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path, device="cpu", model=None, optimizer=None, scheduler=None):
    blob = torch.load(Path(path) / "checkpoint.pt", map_location=device, weights_only=False)
    model = model or build_model(checkpoint_model_config(blob))
    model.load_state_dict(blob["model"])
    if optimizer is not None and "optimizer" in blob: optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None and "scheduler" in blob: scheduler.load_state_dict(blob["scheduler"])
    return model, blob
