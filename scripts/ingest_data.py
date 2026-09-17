#!/usr/bin/env python3
"""ShiftGuard-SecLM: Multi-Source Dataset Ingestion Pipeline.

Combines authentic real-world security corpora (NIST Juliet, CVEFixes,
MITRE CWE Top 25, OWASP Top 10) with controlled multi-archetype synthetic scenarios,
producing canonical raw datasets with complete provenance and licensing attribution.
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.schemas.security_sample import SecuritySample
from src.data.loaders.taxonomy_loader import generate_taxonomy_samples
from src.data.loaders.real_vulnerabilities import generate_real_vulnerability_samples
from src.data.preprocessing.synthetic_generator import SyntheticSecurityGenerator


def run_ingestion(
    num_scenarios_per_archetype: int = 12,
    output_raw: str = "datasets/processed/corpus_raw.jsonl",
    output_source_manifest: str = "datasets/manifests/source_manifest.json",
) -> List[SecuritySample]:
    print("=" * 80)
    print(" SHIFTGUARD-SECLM: MULTI-SOURCE CORPUS INGESTION ENGINE")
    print("=" * 80)

    out_file = Path(output_raw)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(output_source_manifest)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    all_samples: List[SecuritySample] = []
    source_stats: Dict[str, Dict[str, Any]] = {}

    # 1. Ingest Real-World Vulnerability & Fix Corpora (Juliet + CVEFixes)
    print("\n[1/3] Ingesting authentic CVEFixes and NIST Juliet paired vulnerability corpora...")
    real_samples = generate_real_vulnerability_samples()
    all_samples.extend(real_samples)
    print(f"  * Ingested {len(real_samples)} verified real-world vulnerability instances.")

    # 2. Ingest MITRE CWE & OWASP Security Taxonomies
    print("\n[2/3] Ingesting MITRE CWE Top 25 and OWASP Top 10 standard taxonomies...")
    tax_samples = generate_taxonomy_samples()
    all_samples.extend(tax_samples)
    print(f"  * Ingested {len(tax_samples)} canonical taxonomy mapping instances.")

    # 3. Generate Controlled Synthetic Scenarios across 16 Archetypes
    print(f"\n[3/3] Generating synthetic scenarios across 16 archetypes ({num_scenarios_per_archetype} scenarios each)...")
    generator = SyntheticSecurityGenerator(seed=42)
    synth_samples = generator.generate_corpus(num_scenarios_per_archetype=num_scenarios_per_archetype)
    all_samples.extend(synth_samples)
    print(f"  * Generated {len(synth_samples)} structured synthetic instances.")

    # 4. Compile Source Manifest
    for s in all_samples:
        src = s.source
        if src not in source_stats:
            source_stats[src] = {
                "source_name": src,
                "count": 0,
                "license": s.license,
                "is_synthetic": s.provenance.is_synthetic,
                "evidence_sources": set(),
                "languages": set(),
                "archetypes": set(),
            }
        source_stats[src]["count"] += 1
        if s.provenance.evidence_source:
            source_stats[src]["evidence_sources"].add(s.provenance.evidence_source)
        if s.language:
            source_stats[src]["languages"].add(s.language)
        if s.application_archetype:
            source_stats[src]["archetypes"].add(s.application_archetype)

    # Convert sets to lists for JSON serialization
    serialized_manifest = {}
    for k, v in source_stats.items():
        serialized_manifest[k] = {
            "source_name": v["source_name"],
            "count": v["count"],
            "license": v["license"],
            "is_synthetic": v["is_synthetic"],
            "evidence_sources": sorted(list(v["evidence_sources"])),
            "languages": sorted(list(v["languages"])),
            "archetypes": sorted(list(v["archetypes"])),
        }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(serialized_manifest, f, indent=2)

    # 5. Write raw JSONL output
    with open(out_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(sample.model_dump_json() + "\n")

    print("\n--- INGESTION SUMMARY TABLE ---")
    print(f"{'Source':<30} | {'Count':<8} | {'Type':<12} | {'License':<12}")
    print("-" * 70)
    for src, info in serialized_manifest.items():
        type_str = "Synthetic" if info["is_synthetic"] else "Authentic Real"
        print(f"{src:<30} | {info['count']:<8} | {type_str:<12} | {info['license']:<12}")
    print("-" * 70)
    print(f"Total Ingested Raw Records: {len(all_samples)}")
    print(f"Raw Corpus JSONL:           {out_file}")
    print(f"Source Manifest JSON:       {manifest_file}")
    print("=" * 80)

    return all_samples


def main():
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Multi-Source Ingestion")
    parser.add_argument("--num-synthetic-per-archetype", type=int, default=12, help="Number of scenarios per archetype")
    parser.add_argument("--output-raw", type=str, default="datasets/processed/corpus_raw.jsonl", help="Raw output JSONL path")
    parser.add_argument("--output-manifest", type=str, default="datasets/manifests/source_manifest.json", help="Source manifest path")
    args = parser.parse_args()

    run_ingestion(
        num_scenarios_per_archetype=args.num_synthetic_per_archetype,
        output_raw=args.output_raw,
        output_source_manifest=args.output_manifest,
    )


if __name__ == "__main__":
    main()
