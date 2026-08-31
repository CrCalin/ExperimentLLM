"""Registered optimizer and learning-rate schedule factories."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
import math
from typing import Any

import torch

from .config import ComponentSpec
from .registry import OPTIMIZERS, SCHEDULES


class OptimizerFactory(ABC):
    @abstractmethod
    def build(self, parameters: Iterable[torch.nn.Parameter] |
              Iterable[tuple[str, torch.nn.Parameter]]) -> torch.optim.Optimizer: ...


def _named_parameters(parameters) -> list[tuple[str, torch.nn.Parameter]]:
    result = []
    for index, item in enumerate(parameters):
        if isinstance(item, tuple):
            name, parameter = item
        else:
            name, parameter = f"parameter_{index}", item
        result.append((name, parameter))
    return result


@OPTIMIZERS.register
class AdamW(OptimizerFactory):
    def __init__(self, lr: float, betas: list[float] | tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1):
        if lr <= 0: raise ValueError("AdamW lr must be positive")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("AdamW betas must contain two values in [0, 1)")
        if eps <= 0: raise ValueError("AdamW eps must be positive")
        if weight_decay < 0: raise ValueError("AdamW weight_decay cannot be negative")
        self.args = {"lr": lr, "betas": tuple(betas), "eps": eps, "weight_decay": weight_decay}

    def build(self, parameters) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for _, parameter in _named_parameters(parameters):
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 1 or getattr(parameter, "_no_weight_decay", False):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        groups = [
            {"params": decay, "weight_decay": self.args["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, **{key: value for key, value in self.args.items()
                                            if key != "weight_decay"})


class _CompositeOptimizer(torch.optim.Optimizer):
    """One scheduler/checkpoint-compatible facade over Muon and AdamW."""
    def __init__(self, muon: torch.optim.Optimizer, adamw: torch.optim.Optimizer,
                 routing: dict[str, list[str]]):
        self.optimizers = (muon, adamw)
        self.state = {}
        self.param_groups = muon.param_groups + adamw.param_groups
        self.defaults = {}
        self.routing = routing

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("MuonAdamW does not support closures")
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": self.optimizers[0].state_dict(),
            "adamw": self.optimizers[1].state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "muon" not in state_dict or "adamw" not in state_dict:
            raise ValueError("MuonAdamW checkpoint is missing Muon or AdamW optimizer state")
        self.optimizers[0].load_state_dict(state_dict["muon"])
        self.optimizers[1].load_state_dict(state_dict["adamw"])
        self.param_groups = self.optimizers[0].param_groups + self.optimizers[1].param_groups


@OPTIMIZERS.register
class MuonAdamW(OptimizerFactory):
    def __init__(self, muon_lr: float, adamw_lr: float, momentum: float = 0.95,
                 weight_decay: float = 0.1,
                 adamw_betas: list[float] | tuple[float, float] = (0.9, 0.95),
                 adamw_eps: float = 1e-8, nesterov: bool = True,
                 ns_coefficients: list[float] | tuple[float, float, float] = (3.4445, -4.775, 2.0315),
                 eps: float = 1e-7, ns_steps: int = 5,
                 adjust_lr_fn: str | None = None):
        if muon_lr <= 0 or adamw_lr <= 0:
            raise ValueError("MuonAdamW learning rates must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("MuonAdamW momentum must be in [0, 1)")
        if not isinstance(nesterov, bool):
            raise ValueError("MuonAdamW nesterov must be boolean")
        if weight_decay < 0:
            raise ValueError("MuonAdamW weight_decay cannot be negative")
        if len(adamw_betas) != 2 or not all(0 <= beta < 1 for beta in adamw_betas):
            raise ValueError("MuonAdamW adamw_betas must contain two values in [0, 1)")
        if adamw_eps <= 0 or eps <= 0:
            raise ValueError("MuonAdamW eps values must be positive")
        if len(ns_coefficients) != 3:
            raise ValueError("MuonAdamW ns_coefficients must contain three values")
        if not 1 <= ns_steps < 100:
            raise ValueError("MuonAdamW ns_steps must be in [1, 100)")
        if adjust_lr_fn not in {None, "original", "match_rms_adamw"}:
            raise ValueError("MuonAdamW adjust_lr_fn must be null, 'original', or 'match_rms_adamw'")
        self.muon_args = {
            "lr": muon_lr, "momentum": momentum, "nesterov": nesterov,
            "ns_coefficients": tuple(ns_coefficients), "eps": eps,
            "ns_steps": ns_steps, "adjust_lr_fn": adjust_lr_fn,
        }
        self.adamw_args = {
            "lr": adamw_lr, "betas": tuple(adamw_betas), "eps": adamw_eps,
        }
        self.weight_decay = weight_decay

    def build(self, parameters) -> torch.optim.Optimizer:
        muon_parameters, adamw_decay, adamw_no_decay = [], [], []
        routing: dict[str, list[str]] = defaultdict(list)
        for name, parameter in _named_parameters(parameters):
            if not parameter.requires_grad:
                continue
            policy = getattr(parameter, "_optimizer_policy", None)
            if policy == "muon":
                reason = getattr(parameter, "_optimizer_policy_reason", "explicit_muon")
            elif policy == "adamw":
                reason = getattr(parameter, "_optimizer_policy_reason", "explicit_adamw")
            elif name.endswith(".bias") or name == "bias":
                policy, reason = "adamw", "bias"
            elif parameter.ndim != 2:
                policy, reason = "adamw", f"ndim_{parameter.ndim}"
            else:
                policy, reason = "muon", "hidden_matrix"
            routing[reason].append(name)
            if policy == "muon":
                if parameter.ndim != 2:
                    raise ValueError(f"parameter {name!r} is explicitly Muon-routed but has ndim={parameter.ndim}")
                muon_parameters.append(parameter)
            elif parameter.ndim == 1 or getattr(parameter, "_no_weight_decay", False):
                adamw_no_decay.append(parameter)
            else:
                adamw_decay.append(parameter)
        if not muon_parameters:
            raise ValueError("MuonAdamW found no Muon-eligible parameters")
        if not adamw_decay and not adamw_no_decay:
            raise ValueError("MuonAdamW found no AdamW parameters")
        muon_class = getattr(torch.optim, "Muon", None)
        if muon_class is None:
            raise RuntimeError(
                "MuonAdamW requires native torch.optim.Muon; install a PyTorch version that provides it"
            )
        muon = muon_class([{
            "params": muon_parameters, "weight_decay": self.weight_decay,
            "optimizer_name": "muon",
        }], **self.muon_args)
        adamw = torch.optim.AdamW([
            {"params": adamw_decay, "weight_decay": self.weight_decay,
             "optimizer_name": "adamw"},
            {"params": adamw_no_decay, "weight_decay": 0.0,
             "optimizer_name": "adamw"},
        ], **self.adamw_args)
        return _CompositeOptimizer(muon, adamw, dict(routing))


class ScheduleFactory(ABC):
    @abstractmethod
    def build(self, optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LRScheduler: ...


@SCHEDULES.register
class CosineSchedule(ScheduleFactory):
    def __init__(self, warmup_fraction: float = 0.015, final_lr_fraction: float = 0.1):
        if not 0 <= warmup_fraction < 1: raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0 <= final_lr_fraction <= 1: raise ValueError("final_lr_fraction must be in [0, 1]")
        self.warmup_fraction, self.final_lr_fraction = warmup_fraction, final_lr_fraction

    def build(self, optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
        warmup = max(1, int(total_steps * self.warmup_fraction))

        def factor(step: int) -> float:
            if step < warmup: return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return self.final_lr_fraction + (1 - self.final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@SCHEDULES.register
class WSDSchedule(ScheduleFactory):
    def __init__(self, warmup_fraction: float = 0.015, stable_fraction: float = 0.85,
                 final_lr_fraction: float = 0.1):
        if not 0 <= warmup_fraction < 1: raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0 < stable_fraction <= 1: raise ValueError("stable_fraction must be in (0, 1]")
        if warmup_fraction > stable_fraction:
            raise ValueError("warmup_fraction cannot exceed stable_fraction")
        if not 0 <= final_lr_fraction <= 1: raise ValueError("final_lr_fraction must be in [0, 1]")
        self.warmup_fraction = warmup_fraction
        self.stable_fraction = stable_fraction
        self.final_lr_fraction = final_lr_fraction

    def build(self, optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_steps = max(1, int(total_steps * self.warmup_fraction))
        decay_start = min(total_steps - 1, max(warmup_steps, math.ceil(total_steps * self.stable_fraction) - 1))
        decay_steps = max(1, total_steps - decay_start - 1)

        def factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            if step < decay_start:
                return 1.0
            progress = min(1.0, (step - decay_start) / decay_steps)
            return self.final_lr_fraction + (1 - self.final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_optimizer(spec: ComponentSpec, parameters: Iterable[torch.nn.Parameter] |
                    Iterable[tuple[str, torch.nn.Parameter]]) -> torch.optim.Optimizer:
    factory = OPTIMIZERS.create(spec.class_name, **spec.args)
    if not isinstance(factory, OptimizerFactory): raise TypeError(f"{spec.class_name} is not an OptimizerFactory")
    return factory.build(parameters)


def build_schedule(spec: ComponentSpec, optimizer: torch.optim.Optimizer,
                   total_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
    factory = SCHEDULES.create(spec.class_name, **spec.args)
    if not isinstance(factory, ScheduleFactory): raise TypeError(f"{spec.class_name} is not a ScheduleFactory")
    return factory.build(optimizer, total_steps)
