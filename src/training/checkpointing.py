"""Checkpoint Management for ShiftGuard-SecLM.

Handles model saving, loading, and fault-tolerant resumption across
ephemeral training environments (e.g., Kaggle 9-hour runtime limits).
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from src.model.config import ShiftGuardConfig
from src.model.transformer import ShiftGuardTransformer


def save_checkpoint(
    model: ShiftGuardTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    step: int,
    epoch: int,
    loss: float,
    checkpoint_dir: str | Path,
    config: ShiftGuardConfig,
) -> Path:
    """Saves complete model checkpoint with safetensors weights and training state."""
    ckpt_path = Path(checkpoint_dir) / f"step_{step:07d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 1. Save model weights using safetensors (fast zero-copy and secure)
    weights_path = ckpt_path / "model.safetensors"
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(weights_path))

    # 2. Save training optimizer and scheduler state
    train_state = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    torch.save(train_state, ckpt_path / "train_state.pt")

    # 3. Save model configuration
    config.to_yaml(ckpt_path / "config.yaml")

    # 4. Save metadata summary
    meta = {
        "step": step,
        "epoch": epoch,
        "loss": round(loss, 5),
        "model_name": config.model_name,
        "vocab_size": config.vocab_size,
        "n_layers": config.n_layers,
        "d_model": config.d_model,
    }
    with open(ckpt_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return ckpt_path


def load_checkpoint(
    checkpoint_dir: str | Path,
    device: str = "cpu",
    load_training_state: bool = False,
) -> Tuple[ShiftGuardTransformer, ShiftGuardConfig, Optional[Dict[str, Any]]]:
    """Loads model and optional optimizer state from checkpoint."""
    ckpt_path = Path(checkpoint_dir)
    config = ShiftGuardConfig.from_yaml(ckpt_path / "config.yaml")

    model = ShiftGuardTransformer(config)
    weights_path = ckpt_path / "model.safetensors"
    if weights_path.exists():
        state_dict = load_file(str(weights_path), device=device)
        model.load_state_dict(state_dict)
    else:
        # Fallback to PyTorch checkpoint if safetensors not present
        pt_path = ckpt_path / "pytorch_model.bin"
        state_dict = torch.load(pt_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)

    train_state = None
    if load_training_state:
        state_file = ckpt_path / "train_state.pt"
        if state_file.exists():
            train_state = torch.load(state_file, map_location=device)

    return model, config, train_state

