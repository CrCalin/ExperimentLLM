"""Validated JSON experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import json
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ComponentSpec:
    class_name: str
    args: JsonObject = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: JsonObject, path: str) -> "ComponentSpec":
        if not isinstance(value, dict) or not isinstance(value.get("class"), str):
            raise ValueError(f"{path} requires a string 'class'")
        args = value.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"{path}.args must be an object")
        unknown = set(value) - {"class", "args"}
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        return cls(value["class"], dict(args))

    def to_dict(self) -> JsonObject:
        return {"class": self.class_name, "args": self.args}


@dataclass(frozen=True)
class LayerSpec:
    class_name: str
    args: JsonObject

    @classmethod
    def expand(cls, value: JsonObject, index: int) -> list["LayerSpec"]:
        if not isinstance(value, dict) or not isinstance(value.get("class"), str):
            raise ValueError(f"model.layers[{index}] requires a string 'class'")
        args = value.get("args", {})
        repeat = value.get("repeat", 1)
        if not isinstance(args, dict):
            raise ValueError(f"model.layers[{index}].args must be an object")
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError(f"model.layers[{index}].repeat must be a positive integer")
        unknown = set(value) - {"class", "args", "repeat"}
        if unknown:
            raise ValueError(f"unknown keys in model.layers[{index}]: {sorted(unknown)}")
        return [cls(value["class"], dict(args)) for _ in range(repeat)]

    def to_dict(self) -> JsonObject:
        return {"class": self.class_name, "args": self.args}


@dataclass(frozen=True)
class ModelConfig:
    class_name: str
    vocab_size: int
    context_length: int
    d_model: int
    tie_embeddings: bool
    position_encoding: ComponentSpec
    input_embedding: ComponentSpec
    layers: tuple[LayerSpec, ...]
    final_norm: ComponentSpec
    output_head: ComponentSpec
    initialization: ComponentSpec
    num_mtp_heads: int = 1
    mtp_loss_weight: float = 0.1
    mtp_bottleneck_rank: int = 128
    mtp_head_type: str = "low_rank_xielu"
    mtp_position_fraction: float = 0.5
    mtp_active_steps: int = 1000
    mtp_alternating_parity: bool = True

    @classmethod
    def from_dict(cls, value: JsonObject) -> "ModelConfig":
        required = {
            "class", "vocab_size", "context_length", "d_model", "tie_embeddings",
            "position_encoding", "input_embedding", "layers", "final_norm",
            "output_head", "initialization",
        }
        missing = required - set(value)
        unknown = set(value) - required - {"num_mtp_heads", "mtp_loss_weight", "mtp_bottleneck_rank", "mtp_head_type", "mtp_position_fraction", "mtp_active_steps", "mtp_alternating_parity"}
        if missing: raise ValueError(f"missing model keys: {sorted(missing)}")
        if unknown: raise ValueError(f"unknown model keys: {sorted(unknown)}")
        if not isinstance(value["class"], str): raise ValueError("model.class must be a string")
        for name in ("vocab_size", "context_length", "d_model"):
            if not isinstance(value[name], int) or isinstance(value[name], bool):
                raise ValueError(f"model.{name} must be an integer")
            _positive(f"model.{name}", value[name])
        if not isinstance(value["tie_embeddings"], bool):
            raise ValueError("model.tie_embeddings must be boolean")
        if not isinstance(value["layers"], list) or not value["layers"]:
            raise ValueError("model.layers must be a non-empty array")
        layers = tuple(
            layer
            for index, raw in enumerate(value["layers"])
            for layer in LayerSpec.expand(raw, index)
        )
        num_mtp_heads = value.get("num_mtp_heads", 1)
        if not isinstance(num_mtp_heads, int) or isinstance(num_mtp_heads, bool) or not 1 <= num_mtp_heads <= 32:
            raise ValueError("model.num_mtp_heads must be an integer in [1, 32]")
        mtp_loss_weight = value.get("mtp_loss_weight", 0.1)
        if (not isinstance(mtp_loss_weight, (int, float)) or isinstance(mtp_loss_weight, bool)
                or not math.isfinite(mtp_loss_weight) or mtp_loss_weight < 0):
            raise ValueError("model.mtp_loss_weight must be a finite non-negative number")
        mtp_bottleneck_rank = value.get("mtp_bottleneck_rank", 128)
        if (not isinstance(mtp_bottleneck_rank, int) or isinstance(mtp_bottleneck_rank, bool)
                or mtp_bottleneck_rank <= 0):
            raise ValueError("model.mtp_bottleneck_rank must be a positive integer")
        mtp_head_type = value.get("mtp_head_type", "low_rank_xielu")
        if mtp_head_type not in {"low_rank_xielu", "full_width"}:
            raise ValueError("model.mtp_head_type must be low_rank_xielu or full_width")
        mtp_position_fraction = value.get("mtp_position_fraction", 0.5)
        if (not isinstance(mtp_position_fraction, (int, float)) or isinstance(mtp_position_fraction, bool)
                or not math.isfinite(mtp_position_fraction) or not 0 < mtp_position_fraction <= 1):
            raise ValueError("model.mtp_position_fraction must be in (0, 1]")
        mtp_active_steps = value.get("mtp_active_steps", 1000)
        if not isinstance(mtp_active_steps, int) or isinstance(mtp_active_steps, bool) or mtp_active_steps < 0:
            raise ValueError("model.mtp_active_steps must be a non-negative integer")
        mtp_alternating_parity = value.get("mtp_alternating_parity", True)
        if not isinstance(mtp_alternating_parity, bool):
            raise ValueError("model.mtp_alternating_parity must be boolean")
        return cls(
            value["class"], value["vocab_size"], value["context_length"], value["d_model"],
            value["tie_embeddings"],
            ComponentSpec.from_dict(value["position_encoding"], "model.position_encoding"),
            ComponentSpec.from_dict(value["input_embedding"], "model.input_embedding"), layers,
            ComponentSpec.from_dict(value["final_norm"], "model.final_norm"),
            ComponentSpec.from_dict(value["output_head"], "model.output_head"),
            ComponentSpec.from_dict(value["initialization"], "model.initialization"),
            num_mtp_heads, float(mtp_loss_weight), mtp_bottleneck_rank, mtp_head_type,
            float(mtp_position_fraction),
            mtp_active_steps, mtp_alternating_parity,
        )

    def to_dict(self) -> JsonObject:
        return {
            "class": self.class_name, "vocab_size": self.vocab_size,
            "context_length": self.context_length, "d_model": self.d_model,
            "tie_embeddings": self.tie_embeddings,
            "position_encoding": self.position_encoding.to_dict(),
            "input_embedding": self.input_embedding.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
            "final_norm": self.final_norm.to_dict(),
            "output_head": self.output_head.to_dict(),
            "initialization": self.initialization.to_dict(),
            "num_mtp_heads": self.num_mtp_heads,
            "mtp_loss_weight": self.mtp_loss_weight,
            "mtp_bottleneck_rank": self.mtp_bottleneck_rank,
            "mtp_head_type": self.mtp_head_type,
            "mtp_position_fraction": self.mtp_position_fraction,
            "mtp_active_steps": self.mtp_active_steps,
            "mtp_alternating_parity": self.mtp_alternating_parity,
        }


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: ComponentSpec
    schedule: ComponentSpec
    precision: str = "bf16"
    tokens_per_step: int = 131072
    micro_batch_size: int = 2
    grad_clip: float = 1.0
    epochs: int = 1
    seed: int = 12345
    validation_interval_steps: int = 250
    checkpoint_interval_steps: int = 500
    log_interval_steps: int = 50
    max_steps: int | None = None
    max_tokens: int | None = None
    compile: bool = False
    compile_mode: str = "default"

    @classmethod
    def from_dict(cls, value: JsonObject) -> "TrainingConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()} - {"optimizer", "schedule"}
        unknown = set(value) - allowed - {"optimizer", "schedule", "context_length"}
        if unknown: raise ValueError(f"unknown training keys: {sorted(unknown)}")
        if "optimizer" not in value or "schedule" not in value:
            raise ValueError("training requires optimizer and schedule")
        kwargs = {key: value[key] for key in allowed if key in value}
        result = cls(
            optimizer=ComponentSpec.from_dict(value["optimizer"], "training.optimizer"),
            schedule=ComponentSpec.from_dict(value["schedule"], "training.schedule"), **kwargs,
        )
        if result.precision not in {"bf16", "fp32"}: raise ValueError("training.precision must be bf16 or fp32")
        for name in ("tokens_per_step", "micro_batch_size", "epochs", "validation_interval_steps", "checkpoint_interval_steps", "log_interval_steps"):
            if not isinstance(getattr(result, name), int) or isinstance(getattr(result, name), bool):
                raise ValueError(f"training.{name} must be an integer")
            _positive(f"training.{name}", getattr(result, name))
        if not isinstance(result.seed, int) or isinstance(result.seed, bool): raise ValueError("training.seed must be an integer")
        if not isinstance(result.compile, bool): raise ValueError("training.compile must be boolean")
        if result.compile_mode not in {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
            raise ValueError("training.compile_mode is not supported")
        if not isinstance(result.grad_clip, (int, float)) or isinstance(result.grad_clip, bool):
            raise ValueError("training.grad_clip must be numeric")
        _positive("training.grad_clip", result.grad_clip)
        for name in ("max_steps", "max_tokens"):
            item = getattr(result, name)
            if item is not None:
                if not isinstance(item, int) or isinstance(item, bool): raise ValueError(f"training.{name} must be an integer")
                _positive(f"training.{name}", item)
        return result

    def to_dict(self) -> JsonObject:
        value = asdict(self)
        value["optimizer"] = self.optimizer.to_dict()
        value["schedule"] = self.schedule.to_dict()
        return value


@dataclass(frozen=True)
class DataConfig:
    train_bin: str = "train.bin"
    eval_bin: str = "eval.bin"

    def __post_init__(self):
        if not self.train_bin or not isinstance(self.train_bin, str): raise ValueError("data.train_bin must be a non-empty string")
        if not self.eval_bin or not isinstance(self.eval_bin, str): raise ValueError("data.eval_bin must be a non-empty string")


@dataclass(frozen=True)
class EvaluationConfig:
    validation_batches: int = 100

    def __post_init__(self):
        if not isinstance(self.validation_batches, int) or isinstance(self.validation_batches, bool):
            raise ValueError("evaluation.validation_batches must be an integer")
        if self.validation_batches < 0: raise ValueError("evaluation.validation_batches cannot be negative")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    name: str = field(default="experiment", repr=False, compare=False)

    @classmethod
    def from_dict(cls, value: JsonObject) -> "ExperimentConfig":
        unknown = set(value) - {"model", "training", "data", "evaluation"}
        if unknown: raise ValueError(f"unknown experiment keys: {sorted(unknown)}")
        if "model" not in value or "training" not in value:
            raise ValueError("experiment requires model and training")
        training = TrainingConfig.from_dict(value["training"])
        model = ModelConfig.from_dict(value["model"])
        legacy_context = value["training"].get("context_length")
        if legacy_context is not None and legacy_context != model.context_length:
            raise ValueError("training.context_length must match model.context_length")
        return cls(model, training, DataConfig(**value.get("data", {})),
                   EvaluationConfig(**value.get("evaluation", {})))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as f:
            value = json.load(f)
        if not isinstance(value, dict): raise ValueError("experiment config root must be an object")
        from .resolution import resolve_experiment
        return replace(resolve_experiment(cls.from_dict(value)), name=Path(path).stem)

    def to_dict(self) -> JsonObject:
        return {"model": self.model.to_dict(), "training": self.training.to_dict(),
                "data": asdict(self.data), "evaluation": asdict(self.evaluation)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
