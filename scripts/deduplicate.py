#!/usr/bin/env python3
"""ShiftGuard-SecLM: Dataset Deduplication CLI Script.

Executes exact code-hash deduplication and MinHash/LSH near-duplicate
pruning to prevent data leakage and memorization.
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
from src.data.dedup.exact_dedup import ExactDeduplicator
from src.data.dedup.minhash_lsh import MinHashLSH


def deduplicate_dataset(input_file: Path, output_file: Path, lsh_threshold: float = 0.85):
    print("=" * 80)
    print(" SHIFTGUARD-SECLM: DATASET DEDUPLICATION ENGINE")
    print("=" * 80)
    print(f"Reading: {input_file}")

    exact_dedup = ExactDeduplicator()
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=128)

    unique_samples = []
    total_read = 0
    exact_dups = 0
    near_dups = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_read += 1
            data = json.loads(line)
            sample = SecuritySample(**data)

            # 1. Exact / Normalized code and prompt check
            lang = sample.context.get("language", "python")
            if exact_dedup.is_duplicate(code=sample.code, prompt=sample.prompt, language=lang):
                exact_dups += 1
                continue

            # 2. MinHash LSH near-duplicate check on code
            if sample.code:
                if not lsh.insert(sample.sample_id, sample.code):
                    near_dups += 1
                    continue

            unique_samples.append(sample)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in unique_samples:
            f.write(s.model_dump_json() + "\n")

    print(f"\nProcessing Summary:")
    print(f"  * Total Records Read:           {total_read}")
    print(f"  * Exact Duplicates Pruned:      {exact_dups}")
    print(f"  * Near-Duplicates Pruned (LSH): {near_dups}")
    print(f"  * Unique Clean Records Kept:    {len(unique_samples)}")
    print(f"  * Clean Output Saved:           {output_file}")
    print("=" * 80)


def main():
    in_file = Path("datasets/processed/corpus_raw.jsonl")
    out_file = Path("datasets/processed/corpus_deduped.jsonl")
    if not in_file.exists():
        print(f"Error: {in_file} does not exist. Run scripts/prepare_data.py first.")
        sys.exit(1)
    deduplicate_dataset(in_file, out_file)


if __name__ == "__main__":
    main()
