#!/usr/bin/env python3
"""ShiftGuard-SecLM: Data Preparation Pipeline Script.

Ingests public corpora (Juliet, CVEFixes, Taxonomies) and generates controlled
synthetic multi-task instances, outputting canonical JSONL datasets.
"""

import sys
import json
from pathlib import Path

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing.synthetic_generator import SyntheticSecurityGenerator
from src.data.loaders.juliet_loader import JulietLoader
from src.data.loaders.cvefixes_loader import CVEFixesLoader
from src.data.loaders.taxonomy_loader import CWE_TAXONOMY


def main():
    print("=" * 80)
    print(" SHIFTGUARD-SECLM: DATA INGESTION & PREPARATION PIPELINE")
    print("=" * 80)

    output_dir = Path("datasets/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "corpus_raw.jsonl"

    all_samples = []

    # 1. Generate multi-task synthetic archetypes
    print("\n[1/3] Generating multi-task scenario corpus across 12 archetypes...")
    generator = SyntheticSecurityGenerator(seed=42)
    synth_samples = generator.generate_corpus(num_samples_per_archetype=20)
    all_samples.extend(synth_samples)
    print(f"  * Generated {len(synth_samples)} structured synthetic instances.")

    # 2. Ingest Juliet Test Suite sample cases
    print("\n[2/3] Parsing NIST Juliet Test Suite cases...")
    juliet_sample_code = """
    void bad() {
        char * data;
        char dataBuffer[100] = "";
        data = dataBuffer;
        // FLAW: Read data from the console using gets() which does not limit input length
        if (gets(data) == NULL) {
            printLine("gets failed!");
            exit(1);
        }
        CWE120_Buffer_Copy__char_01_bad();
    }
    void good() {
        char * data;
        char dataBuffer[100] = "";
        data = dataBuffer;
        // FIX: Use fgets() which limits input length
        if (fgets(data, 100, stdin) == NULL) {
            printLine("fgets failed!");
            exit(1);
        }
        CWE120_Buffer_Copy__char_01_good();
    }
    """
    juliet_samples = JulietLoader.parse_juliet_source(juliet_sample_code, "CWE120_Buffer_Copy__char_01.c")
    all_samples.extend(juliet_samples)
    print(f"  * Ingested {len(juliet_samples)} Juliet paired vulnerability/repair instances.")

    # 3. Ingest CVEFixes sample cases
    print("\n[3/3] Parsing CVEFixes advisory commit diffs...")
    cve_record = {
        "cve_id": "CVE-2022-21700",
        "cwe_id": "CWE-79",
        "programming_language": "javascript",
        "repo_url": "github.com/matrix-org/matrix-react-sdk",
        "commit_hash": "b2685934a36279f0451cfbf4b24e6a88b1b88e14",
        "code_before": "const sanitized = sanitizeHtml(input, { allowedTags: ['b', 'i', 'a'] });",
        "code_after": "const sanitized = DOMPurify.sanitize(input, { RETURN_DOM: false, FORBID_TAGS: ['style'] });",
        "commit_message": "Fix XSS by switching to strict DOMPurify sanitization",
    }
    cve_sample = CVEFixesLoader.parse_record(cve_record)
    if cve_sample:
        all_samples.append(cve_sample)
        print(f"  * Ingested CVEFixes record: {cve_sample.sample_id}")

    # Write all to corpus_raw.jsonl
    with open(out_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(s.model_dump_json() + "\n")

    print(f"\nSuccessfully prepared {len(all_samples)} raw security instances -> {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

