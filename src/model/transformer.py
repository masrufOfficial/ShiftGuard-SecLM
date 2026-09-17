"""ShiftGuard-SecLM: Decoder-Only Causal Transformer Architecture.

Trained strictly from scratch from random weight initialization.
Incorporates Pre-RMSNorm, RoPE, SwiGLU, Causal SDPA, and Security Context Fusion.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ShiftGuardConfig
from src.model.norm import RMSNorm
from src.model.rope import precompute_freqs_cis
from src.model.attention import CausalSelfAttention
from src.model.mlp import SwiGLU
from src.model.security_context import SecurityContextFusion


class TransformerBlock(nn.Module):
    """Single Pre-Norm Decoder-Only Transformer Block."""

    def __init__(self, config: ShiftGuardConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attention = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dropout=config.dropout,
            use_bias=config.use_bias,
        )
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            use_bias=config.use_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-Norm Attention Residual
        norm_x = self.attn_norm(x)
        attn_out, new_kv_cache = self.attention(norm_x, freqs_cis, mask=mask, kv_cache=kv_cache)
        x = x + attn_out

        # Pre-Norm SwiGLU Residual
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_kv_cache


class ShiftGuardTransformer(nn.Module):
    """Complete Decoder-Only Causal Transformer for ShiftGuard-SecLM."""

    def __init__(self, config: ShiftGuardConfig):
        super().__init__()
        self.config = config

        # Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # Optional Security Context Fusion Layer
        if config.use_scf:
            self.scf = SecurityContextFusion(config.d_model)
        else:
            self.scf = None

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Output Head
        if config.tie_embeddings:
            self.output = None  # Weights shared with tok_embeddings
        else:
            self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute RoPE complex frequencies table
        freqs_cis = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize all weights randomly
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Random weight initialization from normal distribution scaled for residual depth."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # Special scaled initialization for residual projections (Megatron-LM / GPT-2 style)
            if hasattr(module, "_is_residual_proj"):
                std = std / math.sqrt(2 * self.config.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Forward pass.
        
        input_ids: [batch_size, seq_len]
        labels: [batch_size, seq_len] with -100 for ignored / prompt tokens
        segment_ids: Optional [batch_size, seq_len]
        """
        bsz, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, (
            f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.config.max_seq_len})"
        )

        h = self.tok_embeddings(input_ids)

        if self.scf is not None and segment_ids is not None:
            h = self.scf(h, segment_ids)

        freqs_cis = self.freqs_cis[:seq_len]

        new_kv_caches = [] if kv_caches is not None else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = layer(h, freqs_cis, kv_cache=layer_cache)
            if new_kv_caches is not None:
                new_kv_caches.append(new_cache)

        h = self.norm(h)

        # Logits projection (tied vs untied)
        if self.config.tie_embeddings:
            logits = F.linear(h, self.tok_embeddings.weight)
        else:
            logits = self.output(h)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "kv_caches": new_kv_caches,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with greedy / top-p sampling."""
        self.eval()
        cur_ids = input_ids

        for _ in range(max_new_tokens):
            # Crop to max context if needed
            idx_cond = cur_ids if cur_ids.size(1) <= self.config.max_seq_len else cur_ids[:, -self.config.max_seq_len:]
            outputs = self.forward(idx_cond)
            logits = outputs["logits"][:, -1, :]

            if temperature > 0.0:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            cur_ids = torch.cat([cur_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return cur_ids

