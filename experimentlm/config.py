"""Validated JSON experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    @classmethod
    def from_dict(cls, value: JsonObject) -> "ModelConfig":
        if "class" not in value and "n_layers" in value:
            d_model, n_heads = value.get("d_model", 256), value.get("n_heads", 4)
            return cls.from_dict({
                "class": "CausalLM", "vocab_size": value.get("vocab_size", 8192),
                "context_length": value.get("context_length", 2048), "d_model": d_model,
                "tie_embeddings": False,
                "position_encoding": {"class": "RoPE", "args": {"base": value.get("rope_theta", 10000.0)}},
                "input_embedding": {"class": "TokenEmbedding", "args": {}},
                "layers": [{"class": "TransformerBlock", "repeat": value["n_layers"], "args": {
                    "pre_norm": True,
                    "norm": {"class": "RMSNorm", "args": {"eps": 1e-6}},
                    "attention": {"class": "MultiHeadCausalSelfAttention", "args": {
                        "n_heads": n_heads, "head_dim": d_model // n_heads, "bias": False, "dropout": 0.0,
                    }},
                    "mlp": {"class": "SwiGLU", "args": {
                        "d_ff": value.get("d_ff", 704), "bias": False, "dropout": 0.0,
                    }},
                    "residual": {"class": "ParallelResidual", "args": {}},
                }}],
                "final_norm": {"class": "RMSNorm", "args": {"eps": 1e-6}},
                "output_head": {"class": "LinearLMHead", "args": {"bias": False}},
                "initialization": {"class": "GPTStyleInit", "args": {
                    "std": 0.02, "scale_residual_projections": True,
                }},
            })
        required = {
            "class", "vocab_size", "context_length", "d_model", "tie_embeddings",
            "position_encoding", "input_embedding", "layers", "final_norm",
            "output_head", "initialization",
        }
        missing = required - set(value)
        unknown = set(value) - required
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
        return cls(
            value["class"], value["vocab_size"], value["context_length"], value["d_model"],
            value["tie_embeddings"],
            ComponentSpec.from_dict(value["position_encoding"], "model.position_encoding"),
            ComponentSpec.from_dict(value["input_embedding"], "model.input_embedding"), layers,
            ComponentSpec.from_dict(value["final_norm"], "model.final_norm"),
            ComponentSpec.from_dict(value["output_head"], "model.output_head"),
            ComponentSpec.from_dict(value["initialization"], "model.initialization"),
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
class OutputConfig:
    directory: str = "checkpoints/10m"
    log_directory: str = "logs"

    def __post_init__(self):
        if not self.directory or not isinstance(self.directory, str): raise ValueError("output.directory must be a non-empty string")
        if not self.log_directory or not isinstance(self.log_directory, str): raise ValueError("output.log_directory must be a non-empty string")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, value: JsonObject) -> "ExperimentConfig":
        unknown = set(value) - {"model", "training", "data", "evaluation", "output"}
        if unknown: raise ValueError(f"unknown experiment keys: {sorted(unknown)}")
        if "model" not in value or "training" not in value:
            raise ValueError("experiment requires model and training")
        training = TrainingConfig.from_dict(value["training"])
        model = ModelConfig.from_dict(value["model"])
        legacy_context = value["training"].get("context_length")
        if legacy_context is not None and legacy_context != model.context_length:
            raise ValueError("training.context_length must match model.context_length")
        return cls(model, training, DataConfig(**value.get("data", {})),
                   EvaluationConfig(**value.get("evaluation", {})), OutputConfig(**value.get("output", {})))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as f:
            value = json.load(f)
        if not isinstance(value, dict): raise ValueError("experiment config root must be an object")
        from .resolution import resolve_experiment
        return resolve_experiment(cls.from_dict(value))

    def to_dict(self) -> JsonObject:
        return {"model": self.model.to_dict(), "training": self.training.to_dict(),
                "data": asdict(self.data), "evaluation": asdict(self.evaluation), "output": asdict(self.output)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
