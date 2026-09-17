#!/usr/bin/env python3
"""ShiftGuard-SecLM: Multi-Task Dataset Deduplication Engine.

Executes exact code/prompt hash deduplication and MinHash/LSH near-duplicate
pruning with task-awareness to eliminate duplicate samples while preserving
multi-task representations of underlying security cases.
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.schemas.security_sample import SecuritySample
from src.data.dedup.exact_dedup import ExactDeduplicator
from src.data.dedup.minhash_lsh import MinHashLSH


def deduplicate_dataset(
    input_file: Path,
    output_file: Path,
    report_file: Path,
    lsh_threshold: float = 0.85,
):
    print("=" * 80)
    print(" SHIFTGUARD-SECLM: DATASET DEDUPLICATION ENGINE")
    print("=" * 80)
    print(f"Reading: {input_file}")

    exact_dedup = ExactDeduplicator()
    # Separate MinHash LSH index per task to permit multi-task derivations of same scenario
    task_lsh: Dict[str, MinHashLSH] = {}

    unique_samples = []
    total_read = 0
    exact_dups = 0
    near_dups = 0
    dups_by_task: Dict[str, int] = {}

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_read += 1
            data = json.loads(line)
            sample = SecuritySample(**data)

            task_name = sample.task_type.value
            if task_name not in task_lsh:
                task_lsh[task_name] = MinHashLSH(threshold=lsh_threshold, num_perm=128)

            code = sample.source_code or sample.code or ""
            prompt = sample.developer_prompt or sample.prompt or ""
            lang = sample.language or sample.context.get("language", "python")

            # 1. Exact / Normalized code and prompt check within the same task
            task_prefixed_prompt = f"[{task_name}] {prompt}"
            if exact_dedup.is_duplicate(code=code, prompt=task_prefixed_prompt, language=lang):
                exact_dups += 1
                dups_by_task[task_name] = dups_by_task.get(task_name, 0) + 1
                continue

            # 2. MinHash LSH near-duplicate check within the same task
            if code:
                if not task_lsh[task_name].insert(sample.sample_id, code):
                    near_dups += 1
                    dups_by_task[task_name] = dups_by_task.get(task_name, 0) + 1
                    continue

            unique_samples.append(sample)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in unique_samples:
            f.write(s.model_dump_json() + "\n")

    # Generate deduplication report
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "total_records_read": total_read,
        "exact_duplicates_pruned": exact_dups,
        "near_duplicates_pruned_lsh": near_dups,
        "total_duplicates_pruned": exact_dups + near_dups,
        "duplication_rate_pct": round(((exact_dups + near_dups) / max(total_read, 1)) * 100, 2),
        "unique_clean_records_kept": len(unique_samples),
        "pruned_by_task": dups_by_task,
        "lsh_jaccard_threshold": lsh_threshold,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nProcessing Summary:")
    print(f"  * Total Records Read:           {total_read}")
    print(f"  * Exact Duplicates Pruned:      {exact_dups}")
    print(f"  * Near-Duplicates Pruned (LSH): {near_dups}")
    print(f"  * Total Pruned:                 {exact_dups + near_dups} ({report['duplication_rate_pct']}%)")
    print(f"  * Unique Clean Records Kept:    {len(unique_samples)}")
    print(f"  * Clean Output Saved:           {output_file}")
    print(f"  * Deduplication Report Saved:   {report_file}")
    print("=" * 80)

    return report


def main():
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Dataset Deduplication")
    parser.add_argument("--input", type=str, default="datasets/processed/corpus_normalized.jsonl", help="Input normalized JSONL")
    parser.add_argument("--output", type=str, default="datasets/processed/corpus_deduped.jsonl", help="Output deduped JSONL")
    parser.add_argument("--report", type=str, default="datasets/reports/dedup_report.json", help="Deduplication report path")
    parser.add_argument("--lsh-threshold", type=float, default=0.85, help="MinHash LSH Jaccard similarity threshold")
    args = parser.parse_args()

    deduplicate_dataset(
        input_file=Path(args.input),
        output_file=Path(args.output),
        report_file=Path(args.report),
        lsh_threshold=args.lsh_threshold,
    )


if __name__ == "__main__":
    main()
