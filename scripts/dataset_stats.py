#!/usr/bin/env python3
"""ShiftGuard-SecLM: Dataset Statistics & Balancing Analyzer.

Analyzes ShiftGuard-SecCorpus across multiple dimensions:
- Total, code, natural language, and taxonomy token counts
- Task type distribution across all 12 reasoning tasks
- Authentic real vs synthetic ratio and source provenance
- Multi-language, framework, archetype, and CWE/OWASP coverage
- Severe imbalance detection
- Exports machine-readable statistical reports and manifests.
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.schemas.security_sample import SecuritySample


def approximate_token_count(text: str) -> int:
    """Estimates tokens based on subword whitespace/punctuation heuristic (~4 chars per token)."""
    if not text:
        return 0
    # Average byte/subword ratio for code/text is ~3.8 - 4.2 bytes/token
    return max(1, len(text.encode("utf-8")) // 4)


def analyze_corpus(
    splits_dir: str = "datasets/splits",
    stats_dir: str = "datasets/statistics",
    manifests_dir: str = "datasets/manifests",
    reports_dir: str = "datasets/reports",
) -> Dict[str, Any]:
    print("=" * 80)
    print(" SHIFTGUARD-SECLM: DATASET STATISTICS & BALANCING ANALYZER")
    print("=" * 80)

    sp_path = Path(splits_dir)
    stat_path = Path(stats_dir)
    stat_path.mkdir(parents=True, exist_ok=True)
    mf_path = Path(manifests_dir)
    mf_path.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    split_samples: Dict[str, List[SecuritySample]] = {}
    all_samples: List[SecuritySample] = []

    for sp in splits:
        file_path = sp_path / f"{sp}.jsonl"
        samples = []
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        s = SecuritySample(**json.loads(line))
                        samples.append(s)
                        all_samples.append(s)
        split_samples[sp] = samples

    # Token counting across modalities
    total_code_tokens = 0
    total_nl_tokens = 0
    total_taxonomy_tokens = 0

    task_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    lang_counts: Dict[str, int] = {}
    framework_counts: Dict[str, int] = {}
    archetype_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}
    cwe_counts: Dict[str, int] = {}
    owasp_counts: Dict[str, int] = {}
    real_count = 0
    synth_count = 0

    provenance_records = []

    for s in all_samples:
        # Code tokens
        code = s.source_code or s.code or ""
        repair = s.secure_repair or ""
        c_toks = approximate_token_count(code) + approximate_token_count(repair)
        total_code_tokens += c_toks

        # Natural language tokens
        req = s.requirement or ""
        prompt = s.developer_prompt or s.prompt or ""
        threat = s.threat_description or ""
        exp = s.vulnerability_explanation or ""
        strat = " ".join(s.security_strategy or [])
        nl_toks = (
            approximate_token_count(req)
            + approximate_token_count(prompt)
            + approximate_token_count(threat)
            + approximate_token_count(exp)
            + approximate_token_count(strat)
        )
        total_nl_tokens += nl_toks

        # Taxonomy tokens
        cwes = " ".join(s.cwe or [])
        owasps = " ".join(s.owasp_category or [])
        findings = " ".join(f.message for f in s.security_findings)
        tax_toks = (
            approximate_token_count(cwes)
            + approximate_token_count(owasps)
            + approximate_token_count(findings)
        )
        total_taxonomy_tokens += tax_toks

        # Distributions
        t = (s.task_type.value if hasattr(s.task_type, "value") else str(s.task_type)) if s.task_type else str(s.task)
        task_counts[t] = task_counts.get(t, 0) + 1

        src = s.source or "unknown"
        source_counts[src] = source_counts.get(src, 0) + 1

        if s.provenance.is_synthetic:
            synth_count += 1
        else:
            real_count += 1

        lang = (s.language or s.context.get("language", "unknown")).lower()
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        fw = (s.framework or s.context.get("framework", "standard")).lower()
        framework_counts[fw] = framework_counts.get(fw, 0) + 1

        arch = (s.application_archetype or s.context.get("archetype", "unknown")).lower()
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

        sev = (s.severity or "medium").lower()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        for c in s.cwe or []:
            cwe_counts[c] = cwe_counts.get(c, 0) + 1

        for o in s.owasp_category or []:
            owasp_counts[o] = owasp_counts.get(o, 0) + 1

        provenance_records.append({
            "sample_id": s.sample_id,
            "source": s.source,
            "source_id": s.source_id,
            "repository": s.repository,
            "license": s.license,
            "is_synthetic": s.provenance.is_synthetic,
            "split": s.split,
        })

    total_tokens = total_code_tokens + total_nl_tokens + total_taxonomy_tokens

    # Read deduplication & quality reports if available
    dedup_rep_file = Path(reports_dir) / "dedup_report.json"
    dedup_info = {}
    if dedup_rep_file.exists():
        with open(dedup_rep_file, "r", encoding="utf-8") as f:
            dedup_info = json.load(f)

    quality_rep_file = Path(reports_dir) / "quality_report.json"
    quality_info = {}
    if quality_rep_file.exists():
        with open(quality_rep_file, "r", encoding="utf-8") as f:
            quality_info = json.load(f)

    stats = {
        "dataset_name": "ShiftGuard-SecCorpus",
        "version": "2.0.0",
        "total_clean_samples": len(all_samples),
        "total_raw_samples_audited": quality_info.get("total_records_audited", len(all_samples)),
        "duplicates_pruned": dedup_info.get("total_duplicates_pruned", 0),
        "duplication_rate_pct": dedup_info.get("duplication_rate_pct", 0.0),
        "split_counts": {
            "train": len(split_samples["train"]),
            "validation": len(split_samples["val"]),
            "test": len(split_samples["test"]),
        },
        "token_counts": {
            "total_estimated_tokens": total_tokens,
            "code_tokens": total_code_tokens,
            "natural_language_tokens": total_nl_tokens,
            "security_taxonomy_tokens": total_taxonomy_tokens,
            "avg_tokens_per_sample": round(total_tokens / max(len(all_samples), 1), 1),
        },
        "provenance_breakdown": {
            "authentic_real_samples": real_count,
            "controlled_synthetic_samples": synth_count,
            "real_vs_synthetic_ratio": f"{round((real_count / max(len(all_samples), 1)) * 100, 2)}% Real / {round((synth_count / max(len(all_samples), 1)) * 100, 2)}% Synthetic",
            "sources": source_counts,
        },
        "coverage_metrics": {
            "unique_tasks_covered": len(task_counts),
            "unique_languages": len(lang_counts),
            "unique_frameworks": len(framework_counts),
            "unique_archetypes": len(archetype_counts),
            "unique_cwe_families": len(cwe_counts),
            "unique_owasp_categories": len(owasp_counts),
        },
        "distributions": {
            "tasks": task_counts,
            "languages": lang_counts,
            "frameworks": framework_counts,
            "archetypes": archetype_counts,
            "severity": severity_counts,
            "top_cwes": dict(sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True)[:15]),
            "owasp_categories": owasp_counts,
        },
    }

    # 1. Write dataset_statistics.json
    stats_file = stat_path / "dataset_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # 2. Write task_distribution.json
    task_dist_file = stat_path / "task_distribution.json"
    with open(task_dist_file, "w", encoding="utf-8") as f:
        json.dump({
            "task_counts": task_counts,
            "total_samples": len(all_samples),
            "task_percentages": {k: round(v / max(len(all_samples), 1) * 100, 2) for k, v in task_counts.items()},
        }, f, indent=2)

    # 3. Write dataset_manifest.json
    ds_manifest_file = mf_path / "dataset_manifest.json"
    with open(ds_manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": "ShiftGuard-SecCorpus",
            "schema_version": "2.0.0",
            "total_clean_samples": len(all_samples),
            "splits": stats["split_counts"],
            "tokens": stats["token_counts"],
            "coverage": stats["coverage_metrics"],
        }, f, indent=2)

    # 4. Write provenance_manifest.json
    prov_manifest_file = mf_path / "provenance_manifest.json"
    with open(prov_manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(provenance_records),
            "sources": source_counts,
            "sample_provenance_index": provenance_records[:50],  # Sample index
        }, f, indent=2)

    print("\n--- DATASET BALANCING & METRICS REPORT ---")
    print(f"Total Clean Records:        {len(all_samples):,}")
    print(f"Total Estimated Tokens:     {total_tokens:,} ({stats['token_counts']['avg_tokens_per_sample']} avg/sample)")
    print(f"  * Code Tokens:            {total_code_tokens:,}")
    print(f"  * Natural Language:       {total_nl_tokens:,}")
    print(f"  * Security Taxonomy:      {total_taxonomy_tokens:,}")
    print(f"Provenance Distribution:    {stats['provenance_breakdown']['real_vs_synthetic_ratio']}")
    print(f"Task Coverage:              {len(task_counts)} / 12 tasks")
    print(f"Language Coverage:          {len(lang_counts)} languages ({', '.join(list(lang_counts.keys())[:6])}...)")
    print(f"Archetype Coverage:         {len(archetype_counts)} archetypes")
    print(f"Unique CWE Families:        {len(cwe_counts)} CWEs")
    print(f"Unique OWASP Categories:    {len(owasp_counts)} OWASP categories")
    print(f"\nArtifacts Saved:")
    print(f"  * {stats_file}")
    print(f"  * {task_dist_file}")
    print(f"  * {ds_manifest_file}")
    print(f"  * {prov_manifest_file}")
    print("=" * 80)

    return stats


def main():
    analyze_corpus()


if __name__ == "__main__":
    main()
