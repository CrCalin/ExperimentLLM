"""Registered optimizer and learning-rate schedule factories."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
import math

import torch

from .config import ComponentSpec
from .registry import OPTIMIZERS, SCHEDULES


class OptimizerFactory(ABC):
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    @abstractmethod
    def build(self, parameters: Iterable[torch.nn.Parameter] |
              Iterable[tuple[str, torch.nn.Parameter]]) -> torch.optim.Optimizer: ...


def _named_parameters(parameters) -> list[tuple[str, torch.nn.Parameter]]:
    result = []
    seen = set()
    for index, item in enumerate(parameters):
        if isinstance(item, tuple):
            name, parameter = item
        else:
            name, parameter = f"parameter_{index}", item
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
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
                 final_lr_fraction: float = 0.1, decay_fraction: float | None = None,
                 hold_fraction: float = 0.0):
        if not 0 <= warmup_fraction < 1: raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0 < stable_fraction <= 1: raise ValueError("stable_fraction must be in (0, 1]")
        if warmup_fraction > stable_fraction:
            raise ValueError("warmup_fraction cannot exceed stable_fraction")
        if not 0 <= final_lr_fraction <= 1: raise ValueError("final_lr_fraction must be in [0, 1]")
        if decay_fraction is not None and not 0 < decay_fraction <= 1:
            raise ValueError("decay_fraction must be in (0, 1]")
        if not 0 <= hold_fraction < 1:
            raise ValueError("hold_fraction must be in [0, 1)")
        if decay_fraction is not None and stable_fraction + decay_fraction + hold_fraction > 1:
            raise ValueError("stable, decay, and hold fractions cannot exceed 1")
        self.warmup_fraction = warmup_fraction
        self.stable_fraction = stable_fraction
        self.final_lr_fraction = final_lr_fraction
        self.decay_fraction = decay_fraction
        self.hold_fraction = hold_fraction

    def build(self, optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_steps = max(1, int(total_steps * self.warmup_fraction))
        decay_start = min(total_steps - 1, max(warmup_steps, math.ceil(total_steps * self.stable_fraction) - 1))
        if self.decay_fraction is None:
            decay_end = total_steps
        else:
            decay_end = min(total_steps, max(decay_start + 1,
                                             math.ceil(total_steps * (self.stable_fraction + self.decay_fraction))))
        decay_steps = max(1, decay_end - decay_start - 1)

        def factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            if step < decay_start:
                return 1.0
            if step >= decay_end:
                return self.final_lr_fraction
            progress = min(1.0, (step - decay_start) / decay_steps)
            return self.final_lr_fraction + (1 - self.final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_optimizer(spec: ComponentSpec, parameters: Iterable[torch.nn.Parameter] |
                    Iterable[tuple[str, torch.nn.Parameter]]) -> torch.optim.Optimizer:
    factory = OPTIMIZERS.create(spec.class_name, **spec.args)
    if not isinstance(factory, OptimizerFactory): raise TypeError(f"{spec.class_name} is not an OptimizerFactory")
    return factory.build(parameters)


def prepare_model_for_optimizer(spec: ComponentSpec, model: torch.nn.Module) -> torch.nn.Module:
    factory = OPTIMIZERS.create(spec.class_name, **spec.args)
    if not isinstance(factory, OptimizerFactory):
        raise TypeError(f"{spec.class_name} is not an OptimizerFactory")
    return factory.prepare_model(model)


def build_schedule(spec: ComponentSpec, optimizer: torch.optim.Optimizer,
                   total_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
    factory = SCHEDULES.create(spec.class_name, **spec.args)
    if not isinstance(factory, ScheduleFactory): raise TypeError(f"{spec.class_name} is not a ScheduleFactory")
    return factory.build(optimizer, total_steps)
