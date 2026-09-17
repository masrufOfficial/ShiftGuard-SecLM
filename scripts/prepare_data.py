#!/usr/bin/env python3
"""ShiftGuard-SecLM: End-to-End Dataset Preparation Pipeline.

Orchestrates multi-source ingestion, schema validation, task-aware deduplication,
leakage-safe partitioning, balance auditing, and tokenizer building.
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ingest_data import run_ingestion


def run_full_pipeline(num_scenarios_per_archetype: int = 12):
    print("=" * 80)
    print(" STARTING SHIFTGUARD-SECCORPUS END-TO-END REPRODUCIBLE PIPELINE")
    print("=" * 80)

    # 1. Ingestion
    print("\n>>> STEP 1: Multi-Source Ingestion")
    run_ingestion(num_scenarios_per_archetype=num_scenarios_per_archetype)

    # 2. Quality Validation & Normalization
    print("\n>>> STEP 2: Quality Validation & Normalization")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_data.py")], check=True)

    # 3. Task-Aware Exact & MinHash Deduplication
    print("\n>>> STEP 3: Task-Aware Deduplication")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "deduplicate.py")], check=True)

    # 4. Leakage-Safe Cluster Partitioning
    print("\n>>> STEP 4: Leakage-Safe Cluster Partitioning")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "split_data.py")], check=True)

    # 5. Dataset Statistics & Balancing Analyzer
    print("\n>>> STEP 5: Dataset Statistics & Balancing Analyzer")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "dataset_stats.py")], check=True)

    # 6. Domain-Specialized Tokenizer Training (Zero Leakage)
    print("\n>>> STEP 6: Tokenizer Training (Zero Leakage)")
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "build_tokenizer.py"),
        "--train-data",
        "datasets/manifests/train.jsonl"
    ], check=True)

    print("\n" + "=" * 80)
    print(" SHIFTGUARD-SECCORPUS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Data Ingestion Wrapper")
    parser.add_argument("--num-per-archetype", type=int, default=12, help="Number of scenarios per archetype")
    parser.add_argument("--output", type=str, default="datasets/processed/corpus_raw.jsonl", help="Output JSONL path")
    parser.add_argument("--all", action="store_true", help="Run full end-to-end pipeline (ingest, validate, dedup, split, stats, tokenizer)")
    args = parser.parse_args()

    if args.all:
        run_full_pipeline(num_scenarios_per_archetype=args.num_per_archetype)
    else:
        run_ingestion(
            num_scenarios_per_archetype=args.num_per_archetype,
            output_raw=args.output,
        )


if __name__ == "__main__":
    main()
