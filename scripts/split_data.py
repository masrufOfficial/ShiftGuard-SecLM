#!/usr/bin/env python3
"""ShiftGuard-SecLM: Leakage-Safe Dataset Partitioning Script.

Partitions deduplicated security samples into train, validation, and quarantined test splits
with strict repository, vulnerability case, and archetype isolation, generating
SHA-256 manifests and an automated leakage verification report.
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import List

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.schemas.security_sample import SecuritySample
from src.data.split.leakage_split import LeakageSafeSplitter


def run_splitting(
    input_file: str = "datasets/processed/corpus_deduped.jsonl",
    splits_dir: str = "datasets/splits",
    manifests_dir: str = "datasets/manifests",
    report_file: str = "datasets/reports/leakage_report.json",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
):
    print("=" * 80)
    print(" SHIFTGUARD-SECLM: LEAKAGE-SAFE DATASET PARTITIONING ENGINE")
    print("=" * 80)

    in_path = Path(input_file)
    if not in_path.exists():
        print(f"Error: {in_path} does not exist. Run scripts/deduplicate.py first.")
        sys.exit(1)

    samples: List[SecuritySample] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(SecuritySample(**json.loads(line)))

    print(f"Loaded {len(samples)} clean deduplicated instances from {in_path}.")

    splitter = LeakageSafeSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        held_out_archetypes=["iot_backend", "payment_gateway"],
        seed=seed,
    )

    train_set, val_set, test_set = splitter.partition_samples(samples)

    total = len(samples)
    print(f"\nSplit Distribution:")
    print(f"  * Train set:      {len(train_set):4d} samples ({len(train_set)/max(total, 1)*100:.1f}%)")
    print(f"  * Validation set: {len(val_set):4d} samples ({len(val_set)/max(total, 1)*100:.1f}%)")
    print(f"  * Test set:       {len(test_set):4d} samples ({len(test_set)/max(total, 1)*100:.1f}%) [Frozen Quarantine]")

    # Write to datasets/splits/
    sp_dir = Path(splits_dir)
    sp_dir.mkdir(parents=True, exist_ok=True)
    splitter.generate_manifest(train_set, "train", sp_dir)
    splitter.generate_manifest(val_set, "val", sp_dir)
    splitter.generate_manifest(test_set, "test", sp_dir)

    # Write to datasets/manifests/ for training pipeline compatibility
    mf_dir = Path(manifests_dir)
    mf_dir.mkdir(parents=True, exist_ok=True)
    splitter.generate_manifest(train_set, "train", mf_dir)
    splitter.generate_manifest(val_set, "val", mf_dir)
    splitter.generate_manifest(test_set, "test", mf_dir)

    # Audit zero-leakage and write report
    leakage_rep = splitter.audit_leakage(
        train_samples=train_set,
        val_samples=val_set,
        test_samples=test_set,
        output_report=report_file,
    )

    print(f"\nLeakage Audit Status: {'PASSED (ZERO LEAKAGE)' if leakage_rep['zero_leakage_guarantee_passed'] else 'FAILED'}")
    print(f"  * Repository Overlaps:       {len(leakage_rep['repository_overlaps']['train_test'])}")
    print(f"  * Case Identifier Overlaps:  {len(leakage_rep['vulnerability_case_overlaps']['train_test'])}")
    print(f"  * Quarantined Test Samples:  {leakage_rep['test_set_quarantine']['total_test_samples']}")
    print(f"\nSplits Saved:           {sp_dir}/")
    print(f"Manifests Saved:        {mf_dir}/")
    print(f"Leakage Report Saved:   {report_file}")
    print("=" * 80)

    return train_set, val_set, test_set


def main():
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Dataset Partitioning")
    parser.add_argument("--input", type=str, default="datasets/processed/corpus_deduped.jsonl", help="Input deduplicated JSONL")
    parser.add_argument("--splits-dir", type=str, default="datasets/splits", help="Splits directory")
    parser.add_argument("--manifests-dir", type=str, default="datasets/manifests", help="Manifests directory")
    parser.add_argument("--report", type=str, default="datasets/reports/leakage_report.json", help="Leakage report path")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    args = parser.parse_args()

    run_splitting(
        input_file=args.input,
        splits_dir=args.splits_dir,
        manifests_dir=args.manifests_dir,
        report_file=args.report,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
