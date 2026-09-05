"""Expand registered-class defaults into an unambiguous experiment config."""
from __future__ import annotations

from dataclasses import replace

from .config import ComponentSpec, ExperimentConfig, LayerSpec
from .registry import (
    ATTENTIONS, BLOCKS, EMBEDDINGS, HEADS, INITIALIZATIONS, MLPS, MODELS,
    MEMORIES, NORMS, OPTIMIZERS, POSITIONS, RESIDUALS, ROUTINGS, SCHEDULES,
)


def _component(spec: ComponentSpec, registry, injected: set[str] | None = None) -> ComponentSpec:
    return replace(spec, args=registry.resolve_args(spec.class_name, spec.args, injected))


def _layer(layer: LayerSpec) -> LayerSpec:
    args = BLOCKS.resolve_args(layer.class_name, layer.args, {"d_model"})
    if layer.class_name in {"TransformerBlock", "MatrixMemoryTransformerBlock"}:
        mappings = {
            "norm": (NORMS, {"dim"}), "attention": (ATTENTIONS, {"d_model"}),
            "residual": (RESIDUALS, set()),
        }
        for key, (registry, injected) in mappings.items():
            spec = ComponentSpec.from_dict(args[key], f"{layer.class_name}.args.{key}")
            args[key] = _component(spec, registry, injected).to_dict()
        if args["value_routing"] is not None:
            spec = ComponentSpec.from_dict(args["value_routing"], f"{layer.class_name}.args.value_routing")
            args["value_routing"] = _component(spec, ROUTINGS, {"value_dim", "d_model"}).to_dict()
        if args["mlp"] is not None:
            spec = ComponentSpec.from_dict(args["mlp"], f"{layer.class_name}.args.mlp")
            args["mlp"] = _component(spec, MLPS, {"d_model"}).to_dict()
        if layer.class_name == "MatrixMemoryTransformerBlock":
            spec = ComponentSpec.from_dict(args["memory"], f"{layer.class_name}.args.memory")
            args["memory"] = _component(spec, MEMORIES, {"d_model"}).to_dict()
    elif layer.class_name == "FFNOnlyBlock":
        norm_spec = ComponentSpec.from_dict(args["norm"], "FFNOnlyBlock.args.norm")
        mlp_spec = ComponentSpec.from_dict(args["mlp"], "FFNOnlyBlock.args.mlp")
        args["norm"] = _component(norm_spec, NORMS, {"dim"}).to_dict()
        args["mlp"] = _component(mlp_spec, MLPS, {"d_model"}).to_dict()
    return replace(layer, args=args)


def resolve_experiment(config: ExperimentConfig) -> ExperimentConfig:
    model = config.model
    model = replace(
        model,
        position_encoding=_component(model.position_encoding, POSITIONS, {"context_length"}),
        input_embedding=_component(model.input_embedding, EMBEDDINGS, {"vocab_size", "d_model"}),
        layers=tuple(_layer(layer) for layer in model.layers),
        final_norm=_component(model.final_norm, NORMS, {"dim"}),
        output_head=_component(model.output_head, HEADS, {"d_model", "vocab_size"}),
        initialization=_component(model.initialization, INITIALIZATIONS),
    )
    MODELS.get(model.class_name)
    training = replace(
        config.training,
        optimizer=_component(config.training.optimizer, OPTIMIZERS),
        schedule=_component(config.training.schedule, SCHEDULES),
    )
    return replace(config, model=model, training=training)
