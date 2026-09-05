"""Decoder-only language-model architecture."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from ..config import ModelConfig
from ..parameter_policy import set_optimizer_policy
from ..registry import BLOCKS, EMBEDDINGS, HEADS, INITIALIZATIONS, MODELS, NORMS, POSITIONS
from ..components.attention import (DeltaNetLinearAttention, GatedDeltaNet2LinearAttention, GroupedQueryCausalSelfAttention, MultiHeadCausalSelfAttention, SlidingWindowCausalSelfAttention)
from ..components.blocks import FFNOnlyBlock, MatrixMemoryTransformerBlock, TransformerBlock
from ..components.residual import GatedResidual

@MODELS.register
class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = EMBEDDINGS.create(
            config.input_embedding.class_name, config.vocab_size, config.d_model,
            **config.input_embedding.args,
        )
        self.position_encoding = POSITIONS.create(
            config.position_encoding.class_name, config.context_length,
            **config.position_encoding.args,
        )
        self.blocks = nn.ModuleList([
            BLOCKS.create(layer.class_name, config.d_model, **layer.args) for layer in config.layers
        ])
        self.norm_f = NORMS.create(config.final_norm.class_name, config.d_model, **config.final_norm.args)
        self.lm_head = HEADS.create(
            config.output_head.class_name, config.d_model, config.vocab_size,
            **config.output_head.args,
        )
        for parameter in self.wte.parameters():
            set_optimizer_policy(parameter, "adamw", "token_embedding")
        for parameter in self.lm_head.parameters():
            set_optimizer_policy(parameter, "adamw", "output_head")
        if config.tie_embeddings:
            if self.lm_head.weight.shape != self.wte.weight.shape:
                raise ValueError("input embedding and output head weights cannot be tied")
            self.lm_head.weight = self.wte.weight
            set_optimizer_policy(self.wte.weight, "adamw", "tied_embedding_output")
        initializer = INITIALIZATIONS.create(config.initialization.class_name, **config.initialization.args)
        initializer.apply(self, len(self.blocks))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and loss against already-next-token-aligned labels.

        ``labels[:, t]`` is the target for ``input_ids[:, t]``.  Callers
        performing causal language modeling must shift the token stream when
        constructing the inputs/labels pair; the model deliberately does not
        shift labels implicitly.  This is the contract used by ``TokenStream``
        and the trainer.
        """
        if input_ids.size(1) > self.config.context_length: raise ValueError("sequence exceeds context_length")
        x = self.wte(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask, self.position_encoding)
        logits = self.lm_head(self.norm_f(x))
        loss = None if labels is None else F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        return logits, loss

    def forward_matrix_memory(self, x: torch.Tensor,
                              attention_mask: torch.Tensor | None = None,
                              backend: str | None = None):
        """Layer-major hidden forward exposing final per-layer local states."""
        states = []
        for block in self.blocks:
            if isinstance(block, MatrixMemoryTransformerBlock):
                x, state = block.forward_with_memory_state(
                    x, attention_mask, self.position_encoding, backend=backend,
                )
                states.append(state)
            else:
                x = block(x, attention_mask, self.position_encoding)
        return x, tuple(states)

    def forward_cached(self, input_ids: torch.Tensor, cache: dict | None = None):
        """Run a prefix or one decode step while retaining MHA/GDN states."""
        if input_ids.size(1) > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        cache = cache or {"length": 0, "layers": [None] * len(self.blocks)}
        if cache["length"] + input_ids.size(1) > self.config.context_length:
            raise ValueError("cached sequence exceeds context_length; rebuild the cache")
        x = self.wte(input_ids)
        new_layers = []
        for block, layer_cache in zip(self.blocks, cache["layers"]):
            if isinstance(block, FFNOnlyBlock):
                x = block(x, None, self.position_encoding)
                new_layers.append(None)
                continue
            if not isinstance(block, TransformerBlock) or block.pre_norm is not True:
                raise ValueError("cached generation requires pre-norm TransformerBlock layers")
            if isinstance(block, MatrixMemoryTransformerBlock):
                attention_cache = None if layer_cache is None else layer_cache["attention"]
                memory_state = None if layer_cache is None else layer_cache["memory"]
                attn_out, attention_cache = block.attn.forward_cached(
                    block.norm1(x), self.position_encoding, attention_cache,
                )
                gate = getattr(block.residual, "attention_gate", 1.0)
                x, memory_state = block._memory_reference(
                    x + gate * attn_out,
                    block.initial_memory_state(x) if memory_state is None else memory_state,
                )
                new_layers.append({"attention": attention_cache, "memory": memory_state})
                continue
            if isinstance(block.attn, SlidingWindowCausalSelfAttention):
                attn_out, new_cache = block.attn.forward_cached(
                    block.norm1(x), self.position_encoding, layer_cache,
                )
                gate = getattr(block.residual, "attention_gate", 1.0)
                x = x + gate * attn_out
                new_layers.append(new_cache)
            elif isinstance(block.attn, (MultiHeadCausalSelfAttention, GroupedQueryCausalSelfAttention)):
                attn_out, new_cache = block.attn.forward_cached(
                    block.norm1(x), self.position_encoding, layer_cache,
                )
                gate = getattr(block.residual, "attention_gate", 1.0)
                x = x + gate * attn_out
                new_layers.append(new_cache)
            elif isinstance(block.attn, (DeltaNetLinearAttention, GatedDeltaNet2LinearAttention)):
                state = None if layer_cache is None else layer_cache["state"]
                conv_state = None if layer_cache is None else layer_cache["conv_state"]
                attn_out, state, conv_state, value_route = block.attn.forward_cached_with_route(
                    x=block.norm1(x), state=state, conv_state=conv_state,
                )
                gate = getattr(block.residual, "attention_gate", 1.0)
                x = x + gate * attn_out
                x = block.add_value_route(x, value_route, "pre_ffn")
                new_layers.append({"state": state, "conv_state": conv_state})
            else:
                raise ValueError("cached generation requires known attention implementations")
            if block.mlp is not None:
                gate = getattr(block.residual, "mlp_gate", 1.0)
                x = x + gate * block.mlp(block.norm2(x))
            if isinstance(block.attn, (DeltaNetLinearAttention, GatedDeltaNet2LinearAttention)):
                x = block.add_value_route(x, value_route, "post_block")
        logits = self.lm_head(self.norm_f(x))
        new_cache = {"length": cache["length"] + input_ids.size(1), "layers": new_layers}
        return logits, new_cache

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_report(self) -> dict[str, int]:
        block_counts = [sum(p.numel() for p in block.parameters()) for block in self.blocks]
        mha = sum(n for block, n in zip(self.blocks, block_counts)
                  if isinstance(getattr(block, "attn", None), MultiHeadCausalSelfAttention))
        delta = sum(n for block, n in zip(self.blocks, block_counts)
                    if isinstance(getattr(block, "attn", None), (DeltaNetLinearAttention, GatedDeltaNet2LinearAttention)))
        ffn = sum(sum(p.numel() for p in block.mlp.parameters())
                  for block in self.blocks if block.mlp is not None)
        absent = [block for block in self.blocks
                  if hasattr(block, "attn") and block.mlp is None]
        removed = len(absent) * 3 * self.config.d_model * 768
        removed += sum(self.config.d_model + (isinstance(getattr(block, "residual", None), GatedResidual))
                       for block in absent)
        return {
            "total": self.num_parameters,
            "removed_vs_d_ff_768_all_layers": removed,
            "mha_blocks": mha,
            "deltanet_blocks": delta,
            "ffn_remaining": ffn,
            "matrix_memory": sum(sum(p.numel() for p in block.memory.parameters())
                                 for block in self.blocks if hasattr(block, "memory")),
        }


def build_model(config: ModelConfig) -> nn.Module:
    return MODELS.create(config.class_name, config)


ExperimentLM = CausalLM


__all__ = ["CausalLM", "ExperimentLM", "build_model"]
