from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class ShiftGuardConfig:
    """Configuration class for ShiftGuard-SecLM decoder-only Transformer."""
    model_name: str = "ShiftGuard-SecLM-341M"
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = None  # None -> equal to n_heads (Multi-Head Attention)
    d_ffn: Optional[int] = None       # None -> 8/3 * d_model rounded to multiple of 128
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_bias: bool = False
    use_scf: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_ffn is None:
            # SwiGLU 8/3 rule rounded to multiple of 128
            raw_ffn = int(2 * (4 * self.d_model) / 3)
            self.d_ffn = ((raw_ffn + 127) // 128) * 128
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def calculate_parameter_breakdown(self) -> Dict[str, Any]:
        """Calculates exact parameter counts across all architectural sub-components."""
        # 1. Token Embeddings
        embedding_params = self.vocab_size * self.d_model

        # 2. Per-Layer Attention
        q_params = self.d_model * (self.n_heads * self.head_dim) + (self.n_heads * self.head_dim if self.use_bias else 0)
        k_params = self.d_model * (self.n_kv_heads * self.head_dim) + (self.n_kv_heads * self.head_dim if self.use_bias else 0)
        v_params = self.d_model * (self.n_kv_heads * self.head_dim) + (self.n_kv_heads * self.head_dim if self.use_bias else 0)
        o_params = (self.n_heads * self.head_dim) * self.d_model + (self.d_model if self.use_bias else 0)
        attn_per_layer = q_params + k_params + v_params + o_params

        # 3. Per-Layer RMSNorm (input norm + post-attn norm)
        norm_per_layer = 2 * self.d_model

        # 4. Per-Layer SwiGLU (Gate, Up, Down projections)
        gate_params = self.d_model * self.d_ffn + (self.d_ffn if self.use_bias else 0)
        up_params = self.d_model * self.d_ffn + (self.d_ffn if self.use_bias else 0)
        down_params = self.d_ffn * self.d_model + (self.d_model if self.use_bias else 0)
        ffn_per_layer = gate_params + up_params + down_params

        layer_params = attn_per_layer + norm_per_layer + ffn_per_layer
        all_layers_params = layer_params * self.n_layers

        # 5. Final RMSNorm
        final_norm = self.d_model

        # 6. LM Head (if untied)
        lm_head_params = 0 if self.tie_embeddings else (self.vocab_size * self.d_model)

        total_params = embedding_params + all_layers_params + final_norm + lm_head_params

        return {
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "d_ffn": self.d_ffn,
            "tie_embeddings": self.tie_embeddings,
            "embedding_params": embedding_params,
            "attn_per_layer": attn_per_layer,
            "ffn_per_layer": ffn_per_layer,
            "layer_params": layer_params,
            "all_layers_params": all_layers_params,
            "final_norm": final_norm,
            "lm_head_params": lm_head_params,
            "total_params": total_params,
            "total_params_M": round(total_params / 1e6, 2),
        }

    def estimate_compute_and_memory(
        self,
        batch_size: int = 4,
        seq_len: int = 2048,
        precision_bytes: int = 2,  # 2 for fp16/bf16, 4 for fp32
        use_checkpointing: bool = True,
        use_8bit_adam: bool = False,
    ) -> Dict[str, Any]:
        """Calculates theoretical memory requirements and training FLOPs."""
        params = self.calculate_parameter_breakdown()["total_params"]

        # Weights memory
        model_bytes = params * precision_bytes
        grad_bytes = params * precision_bytes

        # Optimizer memory:
        # Standard AdamW: 4 bytes (FP32 master weight) + 4 bytes (first moment) + 4 bytes (second moment) = 12 bytes/param
        # 8-bit AdamW: 2 bytes (FP16 master) + 1 byte + 1 byte = 4 bytes/param
        optimizer_bytes_per_param = 4 if use_8bit_adam else 12
        optimizer_bytes = params * optimizer_bytes_per_param

        # Static memory total
        static_memory_gb = (model_bytes + grad_bytes + optimizer_bytes) / (1024**3)

        # Approximate activation memory per token
        # Standard transformer without checkpointing: ~ (34 * b * s * d * L) bytes
        # With gradient checkpointing: stored activations per block reduced to 1 input per block
        if use_checkpointing:
            activation_bytes = (batch_size * seq_len * self.d_model * precision_bytes) * (self.n_layers + 4)
        else:
            activation_bytes = (34 * batch_size * seq_len * self.d_model * self.n_layers * precision_bytes)
        activation_memory_gb = activation_bytes / (1024**3)

        total_training_memory_gb = static_memory_gb + activation_memory_gb

        # FLOPs: ~6 * params per token for training (2 forward + 4 backward)
        flops_per_token = 6 * params
        step_tokens = batch_size * seq_len
        flops_per_step = flops_per_token * step_tokens

        return {
            "model_memory_gb": round(model_bytes / (1024**3), 2),
            "gradient_memory_gb": round(grad_bytes / (1024**3), 2),
            "optimizer_memory_gb": round(optimizer_bytes / (1024**3), 2),
            "static_memory_gb": round(static_memory_gb, 2),
            "activation_memory_gb": round(activation_memory_gb, 2),
            "total_training_memory_gb": round(total_training_memory_gb, 2),
            "flops_per_token": flops_per_token,
            "flops_per_step": flops_per_step,
            "tokens_per_step": step_tokens,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> ShiftGuardConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def tiny_smoke(cls) -> ShiftGuardConfig:
        return cls(
            model_name="ShiftGuard-SecLM-Tiny",
            vocab_size=16384,
            d_model=256,
            n_layers=6,
            n_heads=8,
            n_kv_heads=8,
            d_ffn=680,
            max_seq_len=1024,
            tie_embeddings=True,
        )

    @classmethod
    def pilot_110m(cls) -> ShiftGuardConfig:
        return cls(
            model_name="ShiftGuard-SecLM-Pilot-110M",
            vocab_size=32000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=12,
            d_ffn=2048,
            max_seq_len=2048,
            tie_embeddings=True,
        )

    @classmethod
    def research_341m(cls) -> ShiftGuardConfig:
        return cls(
            model_name="ShiftGuard-SecLM-341M",
            vocab_size=32000,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            n_kv_heads=16,
            d_ffn=2816,
            max_seq_len=4096,
            tie_embeddings=True,
        )

    @classmethod
    def scaled_528m(cls) -> ShiftGuardConfig:
        return cls(
            model_name="ShiftGuard-SecLM-Scaled-528M",
            vocab_size=32000,
            d_model=1280,
            n_layers=24,
            n_heads=16,
            n_kv_heads=16,
            d_ffn=3584,
            max_seq_len=4096,
            tie_embeddings=True,
        )

    @classmethod
    def research_1b(cls) -> ShiftGuardConfig:
        """Primary ~1B research model with Grouped-Query Attention (993.61M parameters)."""
        return cls(
            model_name="ShiftGuard-SecLM-1B",
            vocab_size=32000,
            d_model=2048,
            n_layers=20,
            n_heads=16,
            n_kv_heads=8,
            d_ffn=5504,
            max_seq_len=4096,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            dropout=0.0,
            tie_embeddings=True,
            use_bias=False,
            use_scf=True,
            initializer_range=0.02,
        )

    @classmethod
    def research_1b_untied(cls) -> ShiftGuardConfig:
        """~1B research model with untied embedding head (1,059.15M parameters)."""
        cfg = cls.research_1b()
        cfg.model_name = "ShiftGuard-SecLM-1B-Untied"
        cfg.tie_embeddings = False
        return cfg

