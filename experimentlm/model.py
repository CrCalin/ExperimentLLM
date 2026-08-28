"""Small pre-norm RoPE/SwiGLU decoder language model."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 2048
    n_layers: int = 7
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 704
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
    def to_dict(self):
        return asdict(self)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__(); self.n_heads = c.n_heads; self.head_dim = c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False); self.out = nn.Linear(c.d_model, c.d_model, bias=False)
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        pos = torch.arange(t, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq); emb = freqs.repeat_interleave(2, dim=-1)
        cos, sin = emb.cos()[None, None].to(x.dtype), emb.sin()[None, None].to(x.dtype)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, -1))

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__(); self.gate = nn.Linear(c.d_model, c.d_ff, bias=False); self.up = nn.Linear(c.d_model, c.d_ff, bias=False); self.down = nn.Linear(c.d_ff, c.d_model, bias=False)
    def forward(self, x): return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__(); self.norm1 = RMSNorm(c.d_model); self.attn = CausalSelfAttention(c); self.norm2 = RMSNorm(c.d_model); self.mlp = SwiGLU(c)
    def forward(self, x): return x + self.attn(self.norm1(x)) + self.mlp(self.norm2(x))

class ExperimentLM(nn.Module):
    def __init__(self, config=None):
        super().__init__(); self.config = config or ModelConfig(); c = self.config
        self.wte = nn.Embedding(c.vocab_size, c.d_model); self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm_f = RMSNorm(c.d_model); self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.apply(self._init_weights)
        scale = 1 / math.sqrt(2 * c.n_layers)
        for block in self.blocks: block.attn.out.weight.data.mul_(scale); block.mlp.down.weight.data.mul_(scale)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)): nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, labels=None):
        if input_ids.size(1) > self.config.context_length: raise ValueError("sequence exceeds context_length")
        x = self.wte(input_ids)
        for block in self.blocks: x = block(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None if labels is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss
    @property
    def num_parameters(self): return sum(p.numel() for p in self.parameters())
