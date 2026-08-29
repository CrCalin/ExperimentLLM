"""Autoregressive generation independent of the command-line interface."""
from __future__ import annotations

import torch


@torch.inference_mode()
def generate(model, input_ids: torch.Tensor, max_new_tokens: int = 100,
             temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
    model.eval()
    output = input_ids
    use_cache = hasattr(model, "forward_cached")
    cache = None
    if use_cache:
        prefix = output[:, -model.config.context_length:]
        logits, cache = model.forward_cached(prefix)
    for _ in range(max_new_tokens):
        if not use_cache:
            logits, _ = model(output[:, -model.config.context_length:])
        logits = logits[:, -1] / max(temperature, 1e-5)
        if top_k:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            next_token = indices.gather(-1, torch.multinomial(torch.softmax(values, -1), 1))
        else:
            next_token = torch.multinomial(torch.softmax(logits, -1), 1)
        output = torch.cat((output, next_token), dim=1)
        if use_cache:
            if output.size(1) > model.config.context_length:
                logits, cache = model.forward_cached(output[:, -model.config.context_length:])
            else:
                logits, cache = model.forward_cached(next_token, cache)
    return output
