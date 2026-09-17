#!/usr/bin/env python3
"""ShiftGuard-SecLM: Model Training CLI Script.

Supports from-scratch training on local CPU (smoke test) and remote GPUs (Kaggle / A100).
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
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from src.model.config import ShiftGuardConfig
from src.model.transformer import ShiftGuardTransformer
from src.training.multitask_train import SecurityJSONLDataset, collate_security_batch, train_epoch
from src.training.checkpointing import save_checkpoint


def train():
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Training Runner")
    parser.add_argument("--config", type=str, default="configs/model/tiny_smoke.yaml", help="Path to model config YAML")
    parser.add_argument("--train-data", type=str, default="datasets/manifests/train.jsonl", help="Train dataset JSONL")
    parser.add_argument("--tokenizer-path", type=str, default="datasets/processed/tokenizer/tokenizer.json", help="Tokenizer JSON path")
    parser.add_argument("--output-dir", type=str, default="experiments/run_01", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    print("=" * 80)
    print(" SHIFTGUARD-SECLM: TRAINING INITIALIZATION")
    print("=" * 80)
    print(f"Device:         {args.device}")
    print(f"Model Config:   {args.config}")
    print(f"Train Data:     {args.train_data}")
    print(f"Tokenizer:      {args.tokenizer_path}")
    print(f"Output Dir:     {args.output_dir}")

    # 1. Load Tokenizer
    tok_path = Path(args.tokenizer_path)
    if not tok_path.exists():
        print(f"Tokenizer not found at {tok_path}. Run scripts/build_tokenizer.py first.")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"Tokenizer loaded (Vocab size: {tokenizer.get_vocab_size()})")

    # 2. Load Model Config & Initialize Weights from Scratch
    config = ShiftGuardConfig.from_yaml(args.config)
    config.vocab_size = tokenizer.get_vocab_size()  # Match tokenizer vocabulary
    print(f"\nInitializing {config.model_name} strictly from RANDOM weights...")
    model = ShiftGuardTransformer(config).to(args.device)

    param_info = config.calculate_parameter_breakdown()
    print(f"Trainable Parameters: {param_info['total_params']:,} ({param_info['total_params_M']}M)")

    # 3. Load Dataset
    data_path = Path(args.train_data)
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Run scripts/split_data.py first.")
        sys.exit(1)

    pad_id = tokenizer.token_to_id("<|pad|>") or 0
    train_dataset = SecurityJSONLDataset(data_path, tokenizer, max_seq_len=config.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_security_batch(b, pad_token_id=pad_id),
    )
    print(f"Loaded {len(train_dataset)} training sequences into DataLoader.")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if "cuda" in args.device else None

    # 5. Training Loop
    print("\nBeginning training loop...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=args.device,
            scaler=scaler,
        )
        print(f"  * Epoch {epoch:2d}/{args.epochs:2d} | Mean Loss: {loss:.4f}")

        # Save checkpoint after each epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            step=epoch * len(train_loader),
            epoch=epoch,
            loss=loss,
            checkpoint_dir=out_dir,
            config=config,
        )

    print(f"\nTraining completed successfully! Checkpoints stored in {out_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    train()

