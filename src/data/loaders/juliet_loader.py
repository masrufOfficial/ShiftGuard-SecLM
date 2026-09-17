"""NIST Juliet Test Suite v1.3 Ingestion Adapter for ShiftGuard-SecLM.

Normalizes Juliet C/C++ and Java synthetic test cases, pairing vulnerable (`bad()`)
and secure (`good()`) reference implementations into canonical SecuritySample records.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets.schemas.security_sample import (
    SecuritySample,
    TaskType,
    ConfidenceLevel,
    SeverityLevel,
    VerificationStatus,
    SecurityFinding,
    SecurityGroundTruth,
    ProvenanceMetadata,
)


class JulietLoader:
    """Parses Juliet Test Suite files and extracts paired vulnerability/repair instances."""

    def __init__(self, base_dir: Optional[str | Path] = None):
        self.base_dir = Path(base_dir) if base_dir else None

    @staticmethod
    def extract_cwe_from_filename(filename: str) -> str:
        """Extracts CWE identifier from Juliet filename conventions (e.g. CWE89_SQL_Injection...)."""
        match = re.search(r"CWE(\d+)", filename, re.IGNORECASE)
        if match:
            return f"CWE-{match.group(1)}"
        return "CWE-Unknown"

    @staticmethod
    def parse_juliet_source(code_text: str, filename: str) -> List[SecuritySample]:
        """Extracts bad() and good() blocks from a Juliet source file."""
        cwe = JulietLoader.extract_cwe_from_filename(filename)
        samples = []

        # Find bad() function block with flexible whitespace
        bad_match = re.search(r"void\s+(?:bad|CWE\w+::bad)\s*\([^)]*\)\s*\{([\s\S]*?)\s*\}", code_text)
        good_match = re.search(r"void\s+(?:good|CWE\w+::good)\s*\([^)]*\)\s*\{([\s\S]*?)\s*\}", code_text)

        if bad_match:
            bad_code = bad_match.group(0)
            repair_code = good_match.group(0) if good_match else None

            sample = SecuritySample(
                sample_id=f"juliet-{Path(filename).stem}",
                task=TaskType.VULN_ANALYSIS,
                requirement=f"Ensure execution prevents vulnerabilities associated with {cwe}.",
                prompt="Analyze the following source implementation for memory or input security flaws.",
                context={
                    "language": "c" if filename.endswith((".c", ".h")) else "cpp" if filename.endswith(".cpp") else "java",
                    "framework": "standard_library",
                    "has_crypto": False,
                },
                code=bad_code,
                security_findings=[
                    SecurityFinding(
                        source="juliet_ground_truth",
                        rule_id=f"juliet.{cwe.lower()}",
                        cwe=cwe,
                        severity=SeverityLevel.HIGH,
                        message=f"Flawed execution flow exhibiting {cwe}.",
                    )
                ],
                ground_truth=SecurityGroundTruth(
                    threats=[f"Vulnerability: {cwe}"],
                    cwe=[cwe],
                    owasp=["A03:2021-Injection" if cwe in ["CWE-89", "CWE-78"] else "A06:2021-Vulnerable and Outdated Components"],
                    repair_guidance=[f"Eliminate unchecked input or invalid pointer operations corresponding to {cwe}."],
                    repair_code=repair_code,
                    verification=VerificationStatus.FIXED if repair_code else None,
                ),
                confidence=ConfidenceLevel.HIGH,
                metadata=ProvenanceMetadata(
                    source_name="juliet_test_suite_v1.3",
                    source_id=filename,
                    repository="nist/juliet",
                    license="CC0-1.0",
                    is_synthetic=True,
                    archetype="controlled_flaw",
                ),
            )
            samples.append(sample)

        return samples
