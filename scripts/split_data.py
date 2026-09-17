#!/usr/bin/env python3
"""ShiftGuard-SecLM: Leakage-Safe Dataset Splitting Script.

Partitions deduplicated security samples into train, validation, and test splits
with strict repository and archetype isolation, generating SHA-256 manifests.
"""

import sys
import json
from pathlib import Path

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.schemas.security_sample import SecuritySample
from src.data.split.leakage_split import LeakageSafeSplitter


def main():
    print("=" * 80)
    print(" SHIFTGUARD-SECLM: LEAKAGE-SAFE DATASET PARTITIONING")
    print("=" * 80)

    in_file = Path("datasets/processed/corpus_deduped.jsonl")
    if not in_file.exists():
        print(f"Error: {in_file} does not exist. Run scripts/deduplicate.py first.")
        sys.exit(1)

    samples = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(SecuritySample(**json.loads(line)))

    print(f"Loaded {len(samples)} deduplicated instances.")

    splitter = LeakageSafeSplitter(
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        held_out_archetypes=["iot_backend", "payment_gateway"],
        seed=42,
    )

    train_set, val_set, test_set = splitter.partition_samples(samples)

    print(f"\nSplit Distribution:")
    print(f"  * Train set:      {len(train_set):4d} samples ({len(train_set)/len(samples)*100:.1f}%)")
    print(f"  * Validation set: {len(val_set):4d} samples ({len(val_set)/len(samples)*100:.1f}%)")
    print(f"  * Test set:       {len(test_set):4d} samples ({len(test_set)/len(samples)*100:.1f}%) [Includes held-out archetypes]")

    out_dir = Path("datasets/processed")
    manifest_dir = Path("datasets/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    splitter.generate_manifest(train_set, "train", manifest_dir)
    splitter.generate_manifest(val_set, "val", manifest_dir)
    splitter.generate_manifest(test_set, "test", manifest_dir)

    print(f"\nCryptographic manifests successfully written to {manifest_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
