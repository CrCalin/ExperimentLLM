"""Explicit component registries used by experiment configuration."""
from __future__ import annotations

import inspect
from typing import TypeVar


T = TypeVar("T")


class Registry:
    def __init__(self, category: str):
        self.category = category
        self._classes: dict[str, type] = {}

    def register(self, cls: type[T]) -> type[T]:
        name = cls.__name__
        if name in self._classes:
            raise ValueError(f"duplicate {self.category} registration: {name}")
        self._classes[name] = cls
        return cls

    def get(self, name: str) -> type:
        try:
            return self._classes[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._classes))
            raise ValueError(f"unknown {self.category} class {name!r}; registered: {choices}") from exc

    def create(self, name: str, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def resolve_args(self, name: str, args: dict, injected: set[str] | None = None) -> dict:
        resolved = dict(args)
        injected = injected or set()
        parameters = list(inspect.signature(self.get(name).__init__).parameters.values())
        accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
        allowed = {p.name for p in parameters if p.name != "self"} - injected
        if not accepts_extra and (unknown := set(resolved) - allowed):
            raise ValueError(f"unknown args for {self.category} class {name}: {sorted(unknown)}")
        for parameter in parameters:
            if parameter.name in {"self", *injected} or parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if parameter.name not in resolved and parameter.default is not inspect.Parameter.empty:
                resolved[parameter.name] = parameter.default
            elif parameter.name not in resolved:
                raise ValueError(f"missing arg {parameter.name!r} for {self.category} class {name}")
        return resolved


MODELS = Registry("model")
EMBEDDINGS = Registry("embedding")
POSITIONS = Registry("position encoding")
BLOCKS = Registry("block")
NORMS = Registry("norm")
ATTENTIONS = Registry("attention")
MLPS = Registry("MLP")
RESIDUALS = Registry("residual")
HEADS = Registry("output head")
INITIALIZATIONS = Registry("initialization")
OPTIMIZERS = Registry("optimizer")
SCHEDULES = Registry("schedule")
