import json
from pathlib import Path
import torch
from .model import ExperimentLM, ModelConfig

def save_checkpoint(path, model, optimizer, scheduler, step, tokens_seen, config):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "tokens_seen": tokens_seen, "config": config.to_dict()}, path / "checkpoint.pt")
    (path / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

def load_checkpoint(path, device="cpu", model=None, optimizer=None, scheduler=None):
    blob = torch.load(Path(path) / "checkpoint.pt", map_location=device, weights_only=False)
    config = ModelConfig.from_dict(blob["config"]); model = model or ExperimentLM(config)
    model.load_state_dict(blob["model"])
    if optimizer is not None and "optimizer" in blob: optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None and "scheduler" in blob: scheduler.load_state_dict(blob["scheduler"])
    return model, blob
