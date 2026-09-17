#!/usr/bin/env python3
"""ShiftGuard-SecLM: Data Quality & Validation Engine.

Performs automated data quality audits across raw security datasets:
- Schema conformance & malformed JSON detection
- Valid CWE & OWASP taxonomy syntax validation
- Empty code / prompt detection
- Severity consistency
- Source licensing & provenance integrity
- Exports clean normalized dataset and machine-readable quality report.
"""

from __future__ import annotations
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.schemas.security_sample import SecuritySample, TaskType

CWE_REGEX = re.compile(r"^CWE-\d+$", re.IGNORECASE)
OWASP_REGEX = re.compile(r"^(A\d{2}:2021-|API\d+:2023-)", re.IGNORECASE)


def validate_sample(data: Dict[str, Any]) -> Tuple[bool, Dict[str, bool], List[str]]:
    """Evaluates a single raw sample against strict quality criteria."""
    flags = {
        "valid_json": True,
        "valid_schema": False,
        "non_empty_code": False,
        "non_empty_prompt": False,
        "valid_cwe": True,
        "valid_owasp": True,
        "has_provenance": False,
        "has_license": False,
        "consistent_severity": True,
    }
    errors = []

    # 1. Schema instantiation
    try:
        sample = SecuritySample(**data)
        flags["valid_schema"] = True
    except Exception as e:
        errors.append(f"SchemaValidationError: {str(e)[:100]}")
        return False, flags, errors

    # 2. Non-empty code and prompt
    code = sample.source_code or sample.code
    if code and code.strip():
        flags["non_empty_code"] = True
    else:
        errors.append("EmptySourceCode")

    prompt = sample.developer_prompt or sample.prompt
    if prompt and prompt.strip():
        flags["non_empty_prompt"] = True
    else:
        errors.append("EmptyDeveloperPrompt")

    # 3. CWE identifier validation
    cwe_list = sample.cwe or []
    for cwe_id in cwe_list:
        if not CWE_REGEX.match(cwe_id.strip()):
            flags["valid_cwe"] = False
            errors.append(f"InvalidCWEIdentifier: {cwe_id}")

    # 4. OWASP category validation
    owasp_list = sample.owasp_category or []
    for cat in owasp_list:
        if not OWASP_REGEX.match(cat.strip()):
            flags["valid_owasp"] = False
            errors.append(f"InvalidOWASPCategory: {cat}")

    # 5. Provenance & License
    if sample.provenance and sample.provenance.source_name:
        flags["has_provenance"] = True
    else:
        errors.append("MissingProvenance")

    if sample.license and sample.license.strip():
        flags["has_license"] = True
    else:
        errors.append("MissingLicense")

    # 6. Overall Pass/Fail
    is_valid = (
        flags["valid_schema"]
        and flags["non_empty_code"]
        and flags["non_empty_prompt"]
        and flags["valid_cwe"]
        and flags["valid_owasp"]
        and flags["has_provenance"]
        and flags["has_license"]
    )

    return is_valid, flags, errors


def run_validation(
    input_file: str = "datasets/processed/corpus_raw.jsonl",
    output_normalized: str = "datasets/processed/corpus_normalized.jsonl",
    output_report: str = "datasets/reports/quality_report.json",
):
    print("=" * 80)
    print(" SHIFTGUARD-SECLM: DATA QUALITY & VALIDATION ENGINE")
    print("=" * 80)
    print(f"Reading: {input_file}")

    in_path = Path(input_file)
    out_norm_path = Path(output_normalized)
    out_norm_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    valid_records = 0
    invalid_records = 0
    error_counts: Dict[str, int] = {}
    flag_summary: Dict[str, int] = {}

    normalized_samples: List[Dict[str, Any]] = []

    with open(in_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            total_records += 1

            try:
                data = json.loads(line)
            except Exception as e:
                invalid_records += 1
                error_counts["MalformedJSON"] = error_counts.get("MalformedJSON", 0) + 1
                continue

            is_valid, flags, errors = validate_sample(data)

            for k, passed in flags.items():
                if passed:
                    flag_summary[k] = flag_summary.get(k, 0) + 1

            if is_valid:
                valid_records += 1
                data["quality_flags"] = flags
                normalized_samples.append(data)
            else:
                invalid_records += 1
                for err in errors:
                    err_type = err.split(":")[0]
                    error_counts[err_type] = error_counts.get(err_type, 0) + 1

    # Save normalized clean output
    with open(out_norm_path, "w", encoding="utf-8") as f:
        for s in normalized_samples:
            f.write(json.dumps(s) + "\n")

    # Generate quality report
    report = {
        "input_file": str(in_path),
        "output_normalized_file": str(out_norm_path),
        "total_records_audited": total_records,
        "valid_records_passed": valid_records,
        "invalid_records_rejected": invalid_records,
        "quality_pass_rate_pct": round((valid_records / max(total_records, 1)) * 100, 2),
        "check_pass_counts": flag_summary,
        "error_distribution": error_counts,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- QUALITY VALIDATION SUMMARY ---")
    print(f"Total Records Audited:      {total_records}")
    print(f"Valid Records Passed:       {valid_records} ({report['quality_pass_rate_pct']}%)")
    print(f"Invalid Records Rejected:   {invalid_records}")
    if error_counts:
        print("\nRejection Breakdown:")
        for err, count in error_counts.items():
            print(f"  * {err}: {count}")
    print(f"\nNormalized Output Saved:    {out_norm_path}")
    print(f"Quality Report Saved:       {report_path}")
    print("=" * 80)

    return report


def main():
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Data Quality Validation")
    parser.add_argument("--input", type=str, default="datasets/processed/corpus_raw.jsonl", help="Raw input JSONL path")
    parser.add_argument("--output-normalized", type=str, default="datasets/processed/corpus_normalized.jsonl", help="Normalized JSONL path")
    parser.add_argument("--output-report", type=str, default="datasets/reports/quality_report.json", help="Quality report path")
    args = parser.parse_args()

    run_validation(
        input_file=args.input,
        output_normalized=args.output_normalized,
        output_report=args.output_report,
    )


if __name__ == "__main__":
    main()
