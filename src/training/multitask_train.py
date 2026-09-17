"""ShiftGuard-SecLM: Multi-Task Supervised Training Engine.

Trains the shared decoder-only Transformer from scratch across all 11 security tasks
with prompt loss masking, mixed precision, and gradient accumulation.
"""

from __future__ import annotations
import math
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from src.model.config import ShiftGuardConfig
from src.model.transformer import ShiftGuardTransformer
from src.model.security_context import SecurityContextPayload, BOUNDARY_TOKENS
from src.training.checkpointing import save_checkpoint


class SecurityJSONLDataset(Dataset):
    """Loads JSONL security records and prepares prompt-masked token sequences."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Tokenizer,
        max_seq_len: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Construct SecurityContextPayload
        gt = sample.get("ground_truth", {})
        payload = SecurityContextPayload(
            task=sample.get("task", "threat_analysis"),
            requirement=sample.get("requirement"),
            prompt=sample.get("prompt"),
            language=sample.get("context", {}).get("language"),
            framework=sample.get("context", {}).get("framework"),
            code=sample.get("code"),
            security_findings=sample.get("security_findings", []),
            project_metadata=sample.get("context", {}),
            target_output=gt,
        )

        prompt_str, completion_str = payload.serialize_full_sequence()

        prompt_enc = self.tokenizer.encode(prompt_str)
        completion_enc = self.tokenizer.encode(completion_str)

        prompt_ids = prompt_enc.ids
        completion_ids = completion_enc.ids

        # Combine IDs
        all_ids = prompt_ids + completion_ids
        if len(all_ids) > self.max_seq_len:
            all_ids = all_ids[: self.max_seq_len]

        # Mask labels: prompt tokens set to -100 so loss is computed ONLY on completion
        prompt_len = min(len(prompt_ids), len(all_ids))
        labels = [-100] * prompt_len + all_ids[prompt_len:]

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_security_batch(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pads variable-length sequences dynamically to the longest in the batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def train_epoch(
    model: ShiftGuardTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        use_amp = scaler is not None and device.startswith("cuda")
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=use_amp):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

    return total_loss / max(len(dataloader), 1)

