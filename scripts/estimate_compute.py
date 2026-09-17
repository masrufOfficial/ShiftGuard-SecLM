#!/usr/bin/env python3
"""ShiftGuard-SecLM: Architecture & Compute Estimation Script

Calculates exact parameter counts, memory footprints, and compute/time requirements
for ShiftGuard-SecLM configurations (from Tiny 8.9M up to Primary 1B) across diverse hardware targets.
"""

import sys
import argparse
from pathlib import Path

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model.config import ShiftGuardConfig


def detect_hardware():
    """Inspects and returns local or cloud compute accelerator profile."""
    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_avail else 0
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only (No discrete CUDA GPU)"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_avail else 0.0
    return {
        "cuda": cuda_avail,
        "device_count": device_count,
        "device_name": device_name,
        "vram_gb": round(vram_gb, 2),
    }


def generate_compute_report():
    hw = detect_hardware()

    configs = [
        ShiftGuardConfig.tiny_smoke(),
        ShiftGuardConfig.pilot_110m(),
        ShiftGuardConfig.research_341m(),
        ShiftGuardConfig.scaled_528m(),
        ShiftGuardConfig.research_1b(),
    ]

    print("=" * 80)
    print(" SHIFTGUARD-SECLM ARCHITECTURE & COMPUTE FEASIBILITY REPORT (RESEARCH V2)")
    print("=" * 80)
    print(f"Host Compute Profile: {hw['device_name']}")
    if hw['cuda']:
        print(f"  * Accelerators: {hw['device_count']}x GPU | Total VRAM: {hw['vram_gb']} GB")
    else:
        print("  * Local CPU execution mode active (Kaggle/Cloud GPUs recommended for training)")
    print()

    for cfg in configs:
        params = cfg.calculate_parameter_breakdown()
        mem = cfg.estimate_compute_and_memory(batch_size=2, seq_len=2048, use_checkpointing=True, use_8bit_adam=True)
        mem_standard = cfg.estimate_compute_and_memory(batch_size=2, seq_len=2048, use_checkpointing=True, use_8bit_adam=False)

        is_primary = " [PRIMARY RESEARCH MODEL]" if "1B" in cfg.model_name else ""
        print(f"> Model: {cfg.model_name}{is_primary}")
        print(f"  * Parameters: {params['total_params_M']}M ({params['total_params']:,} weights)")
        print(f"  * Dimensions: Layers={cfg.n_layers}, Hidden={cfg.d_model}, Heads={cfg.n_heads} (KV_Heads={cfg.n_kv_heads}, HeadDim={cfg.head_dim}), FFN={cfg.d_ffn}")
        print(f"  * Embeddings: {params['embedding_params']:,} (Tied Head: {cfg.tie_embeddings})")
        print(f"  * Per-Layer: Attn={params['attn_per_layer']:,}, FFN={params['ffn_per_layer']:,}, Total={params['layer_params']:,}")
        print(f"  * Transformer Body ({cfg.n_layers} layers): {params['all_layers_params']:,} params")
        print(f"  * VRAM Footprint (SeqLen 2048, Batch 2, Checkpointing):")
        print(f"      - With 8-bit AdamW:   {mem['total_training_memory_gb']:.2f} GB (Static: {mem['static_memory_gb']:.2f} GB, Act: {mem['activation_memory_gb']:.2f} GB)")
        print(f"      - With Standard Adam: {mem_standard['total_training_memory_gb']:.2f} GB (Static: {mem_standard['static_memory_gb']:.2f} GB, Act: {mem_standard['activation_memory_gb']:.2f} GB)")
        print(f"  * FLOPs per token: {mem['flops_per_token']:.2e} FLOPs")
        print()

    print("=" * 80)
    print(" TRAINING DURATION ESTIMATES FOR PRIMARY 1B MODEL (~993.6M PARAMS)")
    print("=" * 80)
    
    cfg_1b = ShiftGuardConfig.research_1b()
    params_1b = cfg_1b.calculate_parameter_breakdown()["total_params"]
    flops_per_tok = 6 * params_1b

    gpus = [
        ("Kaggle 1x NVIDIA T4 (16GB)", 18e12),
        ("Kaggle 2x NVIDIA T4 (DDP, 32GB)", 34e12),
        ("Single RTX 3090 / 4090 (24GB)", 80e12),
        ("Single NVIDIA A100 (80GB)", 140e12),
        ("University 4x A100 Cluster", 520e12),
        ("8x NVIDIA A100 Cluster", 1040e12),
    ]

    token_budgets = [
        ("Pilot Target", 1.0),
        ("Intermediate Target", 4.0),
        ("Primary Research Target", 8.0),
        ("Chinchilla Compute-Optimal", 20.0),
    ]

    for label, tok_b in token_budgets:
        total_flops = flops_per_tok * (tok_b * 1e9)
        print(f"\n> Token Budget: {tok_b:.1f} Billion Tokens ({label}) -> {total_flops:.2e} Total FLOPs")
        for gpu_name, eff_flops in gpus:
            hrs = (total_flops / eff_flops) / 3600
            days = hrs / 24
            if hrs < 24:
                print(f"   - {gpu_name:35s}: {hrs:5.1f} hours")
            else:
                print(f"   - {gpu_name:35s}: {hrs:5.1f} hours ({days:4.1f} days)")

    print()
    print("=" * 80)


if __name__ == "__main__":
    generate_compute_report()
