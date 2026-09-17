import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.model.rope import apply_rotary_emb


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention with Rotary Position Embeddings (RoPE) and Grouped-Query Attention (GQA).
    
    Uses PyTorch Scaled Dot-Product Attention (SDPA) for hardware-optimized kernel dispatch
    (FlashAttention-2 / Memory-Efficient SDPA / Math backend).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=use_bias)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=use_bias)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=use_bias)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=use_bias)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for Grouped-Query Attention if n_kv_heads < n_heads."""
        if n_rep == 1:
            return x
        bs, slen, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape

        # Projections: [B, S, H * D] -> [B, S, H, D]
        xq = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # Slice freqs_cis to match current sequence length
        freqs_cis_seq = freqs_cis[:seq_len]
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis_seq)

        # KV Cache handling for fast autoregressive generation
        new_kv_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=1)
            xv = torch.cat([v_cache, xv], dim=1)
            new_kv_cache = (xk, xv)

        # Expand KV heads if using GQA
        xk = self._repeat_kv(xk, self.num_rep)
        xv = self._repeat_kv(xv, self.num_rep)

        # Reshape for SDPA: [B, H, S, D]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Efficient causal attention via SDPA
        is_causal = mask is None and seq_len > 1 and kv_cache is None
        attn_out = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Transpose back: [B, H, S, D] -> [B, S, H * D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(attn_out))

        return output, new_kv_cache

