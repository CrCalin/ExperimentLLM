"""Model evaluation independent of the training loop."""
from __future__ import annotations

import math
import torch


@torch.inference_mode()
def evaluate_loss(model, batches, device: torch.device, max_batches: int | None,
                  precision: str = "bf16") -> tuple[float, float, int]:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for index, (inputs, labels) in enumerate(batches):
        if max_batches is not None and index >= max_batches: break
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == "bf16"):
            _, loss = model(inputs, labels)
        tokens = labels.numel()
        total_loss += float(loss) * tokens
        total_tokens += tokens
    model.train()
    loss = total_loss / max(total_tokens, 1)
    return loss, math.exp(min(loss, 20)), total_tokens
