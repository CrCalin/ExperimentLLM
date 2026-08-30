"""Optimizer-policy annotations shared by model components and optimizer factories."""
from __future__ import annotations

import torch


def set_optimizer_policy(parameter: torch.nn.Parameter, optimizer: str, reason: str) -> None:
    if optimizer not in {"muon", "adamw"}:
        raise ValueError("optimizer policy must be 'muon' or 'adamw'")
    parameter._optimizer_policy = optimizer
    parameter._optimizer_policy_reason = reason
