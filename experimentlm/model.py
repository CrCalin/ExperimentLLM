"""Compatibility imports for the modular model implementation."""
from .components.base import Attention, Block, Memory, MLP, PositionEncoding, Residual
from .components.memory import MatrixMemory
from .components.embeddings import TokenEmbedding
from .components.norms import RMSNorm
from .components.positional import RoPE
from .components.attention import (DeltaNetLinearAttention, GatedDeltaNet2LinearAttention, GroupedQueryCausalSelfAttention, GroupedQuerySlidingWindowCausalSelfAttention, MultiHeadCausalSelfAttention, SlidingWindowCausalSelfAttention)
from .components.attention.common import GatedRMSNorm, _scaled_dot_product_attention
from .components.mlp import SwiGLU
from .components.residual import GatedResidual, StandardResidual
from .components.blocks import FFNOnlyBlock, MatrixMemoryTransformerBlock, TransformerBlock
from .components.heads import LinearLMHead
from .components.initialization import GPTStyleInit
from .architectures.causal_lm import CausalLM, ExperimentLM, build_model

__all__ = ["Attention", "Block", "Memory", "MLP", "PositionEncoding", "Residual", "MatrixMemory", "RMSNorm", "TokenEmbedding", "RoPE", "MultiHeadCausalSelfAttention", "SlidingWindowCausalSelfAttention", "GroupedQueryCausalSelfAttention", "GroupedQuerySlidingWindowCausalSelfAttention", "DeltaNetLinearAttention", "GatedDeltaNet2LinearAttention", "GatedRMSNorm", "SwiGLU", "StandardResidual", "GatedResidual", "TransformerBlock", "MatrixMemoryTransformerBlock", "FFNOnlyBlock", "LinearLMHead", "GPTStyleInit", "CausalLM", "ExperimentLM", "build_model"]
