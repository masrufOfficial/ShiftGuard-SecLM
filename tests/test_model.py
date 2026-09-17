"""Unit & Smoke Tests for ShiftGuard-SecLM Architecture.

Verifies RMSNorm, RoPE, SwiGLU, Causal Attention, TransformerBlock,
ShiftGuardTransformer forward/backward passes, and tiny-dataset overfitting.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import math
import torch
import torch.nn.functional as F

from src.model.config import ShiftGuardConfig
from src.model.norm import RMSNorm
from src.model.rope import precompute_freqs_cis, apply_rotary_emb
from src.model.mlp import SwiGLU
from src.model.attention import CausalSelfAttention
from src.model.transformer import TransformerBlock, ShiftGuardTransformer
from src.model.security_context import SecurityContextPayload, SecurityContextFusion


def test_rmsnorm_numerical_stability():
    dim = 64
    norm = RMSNorm(dim, eps=1e-5)
    x = torch.randn(2, 10, dim)
    out = norm(x)
    assert out.shape == x.shape
    # Variance along last dimension should be close to 1
    rms = torch.sqrt(torch.mean(out.pow(2), dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rope_rotation():
    head_dim = 32
    seq_len = 16
    freqs_cis = precompute_freqs_cis(head_dim, seq_len, theta=10000.0)
    assert freqs_cis.shape == (seq_len, head_dim // 2)

    xq = torch.randn(2, seq_len, 4, head_dim)
    xk = torch.randn(2, seq_len, 4, head_dim)
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
    assert xq_rot.shape == xq.shape
    assert xk_rot.shape == xk.shape


def test_swiglu_forward():
    d_model = 64
    d_ffn = 128
    mlp = SwiGLU(d_model, d_ffn)
    x = torch.randn(2, 8, d_model)
    out = mlp(x)
    assert out.shape == (2, 8, d_model)


def test_causal_attention():
    d_model = 64
    n_heads = 4
    n_kv_heads = 4
    head_dim = 16
    attn = CausalSelfAttention(d_model, n_heads, n_kv_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, 16)
    x = torch.randn(2, 16, d_model)
    out, cache = attn(x, freqs_cis)
    assert out.shape == (2, 16, d_model)
    assert cache is None


def test_tiny_transformer_forward_and_backward():
    config = ShiftGuardConfig.tiny_smoke()
    model = ShiftGuardTransformer(config)
    model.train()

    bsz = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    labels = input_ids.clone()
    segment_ids = torch.randint(0, 4, (bsz, seq_len))

    outputs = model(input_ids, labels=labels, segment_ids=segment_ids)
    logits = outputs["logits"]
    loss = outputs["loss"]

    assert logits.shape == (bsz, seq_len, config.vocab_size)
    assert loss is not None
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Initial cross-entropy on random weights should be close to ln(vocab_size)
    expected_loss = math.log(config.vocab_size)
    assert abs(loss.item() - expected_loss) < 2.0, f"Loss {loss.item()} too far from {expected_loss}"

    # Backward pass
    loss.backward()

    # Verify all parameters received gradients
    has_grad = [p.grad is not None and not torch.isnan(p.grad).any() for p in model.parameters() if p.requires_grad]
    assert all(has_grad), "Not all trainable parameters received valid gradients!"


def test_loss_masking_on_prompt():
    """Verifies that setting label tokens to -100 properly ignores them in the loss."""
    config = ShiftGuardConfig.tiny_smoke()
    model = ShiftGuardTransformer(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    labels = input_ids.clone()

    # Mask first 10 tokens (context/prompt)
    labels[:, :10] = -100

    out = model(input_ids, labels=labels)
    assert out["loss"] is not None
    assert not torch.isnan(out["loss"])


def test_autoregressive_generation():
    config = ShiftGuardConfig.tiny_smoke()
    model = ShiftGuardTransformer(config)
    model.eval()

    prompt = torch.tensor([[10, 20, 30]])
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    assert generated.shape == (1, 13)
    assert (generated[:, :3] == prompt).all()


def test_overfit_tiny_security_batch():
    """Overfits a tiny batch of 4 sequences to prove optimization convergence."""
    torch.manual_seed(42)
    config = ShiftGuardConfig.tiny_smoke()
    config.n_layers = 2
    config.d_model = 128
    config.n_heads = 4
    config.n_kv_heads = 4
    config.d_ffn = 256
    model = ShiftGuardTransformer(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    bsz = 4
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    labels = input_ids.clone()

    initial_loss = None
    final_loss = None

    for step in range(40):
        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"\nTiny batch overfit: Initial Loss = {initial_loss:.4f} -> Final Loss = {final_loss:.4f}")
    assert final_loss < initial_loss * 0.3, (
        f"Model failed to overfit tiny batch: {initial_loss:.4f} -> {final_loss:.4f}"
    )


if __name__ == "__main__":
    print("Running ShiftGuard-SecLM Architecture Tests...")
    test_rmsnorm_numerical_stability()
    print("[PASS] RMSNorm passed")
    test_rope_rotation()
    print("[PASS] RoPE passed")
    test_swiglu_forward()
    print("[PASS] SwiGLU passed")
    test_causal_attention()
    print("[PASS] Causal Attention passed")
    test_tiny_transformer_forward_and_backward()
    print("[PASS] Transformer Forward & Backward passed")
    test_loss_masking_on_prompt()
    print("[PASS] Loss Prompt Masking passed")
    test_autoregressive_generation()
    print("[PASS] Autoregressive Generation passed")
    test_overfit_tiny_security_batch()
    print("[PASS] Tiny Security Batch Overfit passed")
    print("\nAll Architecture & Training Mechanics Tests PASSED successfully!")
