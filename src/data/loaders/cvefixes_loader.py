"""CVEFixes Dataset Ingestion Adapter for ShiftGuard-SecLM.

Extracts real-world CVE-level commits, pre-patch vulnerable code,
and post-patch repairs across open-source repositories.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
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


class CVEFixesLoader:
    """Transforms CVEFixes raw JSON/CSV records into SecuritySample instances."""

    @staticmethod
    def parse_record(record: Dict[str, Any]) -> Optional[SecuritySample]:
        cve_id = record.get("cve_id", "CVE-UNKNOWN")
        cwe_id = record.get("cwe_id", "CWE-UNKNOWN")
        language = record.get("programming_language", "python").lower()
        repo = record.get("repo_url", "unknown_repo")
        commit_hash = record.get("commit_hash", "")
        vuln_code = record.get("code_before", "")
        fixed_code = record.get("code_after", "")
        commit_msg = record.get("commit_message", "")

        if not vuln_code:
            return None

        # Clean CWE formatting (e.g. CWE-89)
        if not cwe_id.startswith("CWE-") and cwe_id.isdigit():
            cwe_id = f"CWE-{cwe_id}"

        return SecuritySample(
            sample_id=f"cvefixes-{cve_id}-{commit_hash[:8]}",
            task=TaskType.REPAIR,
            requirement=f"Remediate {cve_id} in {repo}.",
            prompt=f"Identify and patch the security flaw in this function. Fix details: {commit_msg}",
            context={
                "language": language,
                "framework": record.get("framework", "unknown"),
                "cve": cve_id,
            },
            code=vuln_code,
            security_findings=[
                SecurityFinding(
                    source="cvefixes_git_audit",
                    rule_id=f"cvefixes.{cve_id.lower()}",
                    cwe=cwe_id if cwe_id != "CWE-UNKNOWN" else None,
                    severity=SeverityLevel.HIGH,
                    message=f"Reported security vulnerability {cve_id}.",
                )
            ],
            ground_truth=SecurityGroundTruth(
                threats=[f"Security vulnerability fixed by {commit_hash[:8]}"],
                cwe=[cwe_id] if cwe_id != "CWE-UNKNOWN" else [],
                owasp=["A06:2021-Vulnerable and Outdated Components"],
                repair_guidance=[commit_msg or f"Apply secure patch for {cve_id}."],
                repair_code=fixed_code if fixed_code else None,
                verification=VerificationStatus.FIXED if fixed_code else None,
            ),
            confidence=ConfidenceLevel.HIGH,
            metadata=ProvenanceMetadata(
                source_name="cvefixes",
                source_id=cve_id,
                repository=repo,
                commit_hash=commit_hash,
                license="CC-BY-4.0",
                is_synthetic=False,
            ),
        )
