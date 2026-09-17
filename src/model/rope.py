import torch
import torch.nn as nn
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes frequency complex exponential angles for Rotary Positional Embeddings (RoPE).
    
    dim: head_dim (must be even)
    end: max sequence length
    theta: base frequency
    """
    assert dim % 2 == 0, f"Dimension {dim} must be divisible by 2 for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Convert polar coordinates to complex tensor (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to query and key states.
    
    xq: [batch_size, seq_len, n_heads, head_dim]
    xk: [batch_size, seq_len, n_kv_heads, head_dim]
    freqs_cis: [seq_len, head_dim // 2]
    """
    # View as complex numbers: [batch_size, seq_len, heads, head_dim // 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis across batch and heads: [1, seq_len, 1, head_dim // 2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # Rotate in complex domain
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

