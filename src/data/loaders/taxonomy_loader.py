"""Security Taxonomy Loader: MITRE CWE Top 25 & OWASP Knowledge Base.

Provides structured mapping between CWE vulnerabilities, OWASP Top 10 (2021),
OWASP API Security Top 10 (2023), attack patterns, and standard mitigations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from datasets.schemas.security_sample import (
    SecuritySample,
    TaskType,
    ConfidenceLevel,
    SeverityLevel,
    SecurityGroundTruth,
    ProvenanceMetadata,
)

# Canonical CWE to OWASP 2021 & API 2023 mapping
CWE_TO_OWASP_MAP: Dict[str, str] = {
    "CWE-89": "A03:2021-Injection",
    "CWE-78": "A03:2021-Injection",
    "CWE-79": "A03:2021-Injection",
    "CWE-20": "A03:2021-Injection",
    "CWE-611": "A05:2021-Security Misconfiguration",
    "CWE-287": "A07:2021-Identification and Authentication Failures",
    "CWE-384": "A07:2021-Identification and Authentication Failures",
    "CWE-798": "A07:2021-Identification and Authentication Failures",
    "CWE-306": "A07:2021-Identification and Authentication Failures",
    "CWE-522": "A07:2021-Identification and Authentication Failures",
    "CWE-200": "A01:2021-Broken Access Control",
    "CWE-22": "A01:2021-Broken Access Control",
    "CWE-639": "A01:2021-Broken Access Control",
    "CWE-862": "A01:2021-Broken Access Control",
    "CWE-276": "A01:2021-Broken Access Control",
    "CWE-732": "A01:2021-Broken Access Control",
    "CWE-311": "A02:2021-Cryptographic Failures",
    "CWE-327": "A02:2021-Cryptographic Failures",
    "CWE-328": "A02:2021-Cryptographic Failures",
    "CWE-16": "A05:2021-Security Misconfiguration",
    "CWE-770": "A04:2021-Insecure Design",
    "CWE-502": "A08:2021-Software and Data Integrity Failures",
    "CWE-918": "A10:2021-Server-Side Request Forgery",
    "CWE-119": "A06:2021-Vulnerable and Outdated Components",
    "CWE-120": "A06:2021-Vulnerable and Outdated Components",
    "CWE-787": "A06:2021-Vulnerable and Outdated Components",
    "CWE-125": "A06:2021-Vulnerable and Outdated Components",
    "CWE-416": "A06:2021-Vulnerable and Outdated Components",
    "CWE-476": "A06:2021-Vulnerable and Outdated Components",
    "CWE-190": "A06:2021-Vulnerable and Outdated Components",
    "CWE-352": "A01:2021-Broken Access Control",
    "CWE-434": "A04:2021-Insecure Design",
}

# Canonical CWE Descriptions and Standard Mitigations
CWE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "CWE-89": {
        "name": "SQL Injection",
        "severity": "critical",
        "mitigation": "Use parameterized queries or Object-Relational Mapping (ORM) parameter binding.",
        "owasp": "A03:2021-Injection",
    },
    "CWE-78": {
        "name": "OS Command Injection",
        "severity": "critical",
        "mitigation": "Avoid shell=True; use subprocess with explicit argument arrays and strict whitelist input validation.",
        "owasp": "A03:2021-Injection",
    },
    "CWE-79": {
        "name": "Cross-Site Scripting (XSS)",
        "severity": "high",
        "mitigation": "Context-aware HTML escaping, DOMPurify sanitization, and Content Security Policy (CSP) headers.",
        "owasp": "A03:2021-Injection",
    },
    "CWE-22": {
        "name": "Path Traversal",
        "severity": "high",
        "mitigation": "Resolve canonical path via os.path.realpath and enforce directory root boundary checks.",
        "owasp": "A01:2021-Broken Access Control",
    },
    "CWE-287": {
        "name": "Improper Authentication",
        "severity": "critical",
        "mitigation": "Enforce cryptographic slow password hashing (Argon2id/bcrypt) and multi-factor authentication.",
        "owasp": "A07:2021-Identification and Authentication Failures",
    },
    "CWE-798": {
        "name": "Hardcoded Credentials",
        "severity": "high",
        "mitigation": "Extract credentials into encrypted secret managers or environment variables.",
        "owasp": "A07:2021-Identification and Authentication Failures",
    },
    "CWE-327": {
        "name": "Use of Broken Cryptographic Algorithm",
        "severity": "high",
        "mitigation": "Replace MD5/SHA-1 with SHA-256/SHA-512 and DES with AES-256-GCM.",
        "owasp": "A02:2021-Cryptographic Failures",
    },
    "CWE-918": {
        "name": "Server-Side Request Forgery (SSRF)",
        "severity": "high",
        "mitigation": "Enforce strict IP/domain whitelist; block RFC 1918 private IP ranges and cloud metadata (169.254.169.254).",
        "owasp": "A10:2021-Server-Side Request Forgery",
    },
    "CWE-770": {
        "name": "Allocation of Resources Without Limits (Rate Limiting)",
        "severity": "medium",
        "mitigation": "Implement rate limiting, token bucket algorithms, and payload maximum body size restrictions.",
        "owasp": "A04:2021-Insecure Design",
    },
    "CWE-502": {
        "name": "Deserialization of Untrusted Data",
        "severity": "critical",
        "mitigation": "Avoid pickle/eval/yaml.load; use safe serializers such as json or pydantic with strict schema typing.",
        "owasp": "A08:2021-Software and Data Integrity Failures",
    },
    "CWE-862": {
        "name": "Missing Authorization (BOLA/IDOR)",
        "severity": "critical",
        "mitigation": "Verify tenant and user resource ownership at the data-access layer for every request.",
        "owasp": "A01:2021-Broken Access Control",
    },
    "CWE-787": {
        "name": "Out-of-bounds Write (Buffer Overflow)",
        "severity": "critical",
        "mitigation": "Use memory-safe languages or bound-checked buffer copy primitives (e.g. strlcpy, fgets, std::string).",
        "owasp": "A06:2021-Vulnerable and Outdated Components",
    },
    "CWE-125": {
        "name": "Out-of-bounds Read",
        "severity": "high",
        "mitigation": "Validate bounds against array/buffer length before indexing memory.",
        "owasp": "A06:2021-Vulnerable and Outdated Components",
    },
    "CWE-416": {
        "name": "Use After Free",
        "severity": "critical",
        "mitigation": "Set pointers to NULL immediately upon deallocation; leverage RAII / smart pointers.",
        "owasp": "A06:2021-Vulnerable and Outdated Components",
    },
    "CWE-476": {
        "name": "NULL Pointer Dereference",
        "severity": "medium",
        "mitigation": "Assert non-null status before pointer dereferencing; utilize Optional types.",
        "owasp": "A06:2021-Vulnerable and Outdated Components",
    },
    "CWE-190": {
        "name": "Integer Overflow or Wraparound",
        "severity": "high",
        "mitigation": "Check arithmetic bounds before multiplication or addition, or use overflow-safe numeric libraries.",
        "owasp": "A06:2021-Vulnerable and Outdated Components",
    },
    "CWE-352": {
        "name": "Cross-Site Request Forgery (CSRF)",
        "severity": "high",
        "mitigation": "Require anti-CSRF synchronizer tokens and configure SameSite=Strict cookies.",
        "owasp": "A01:2021-Broken Access Control",
    },
    "CWE-434": {
        "name": "Unrestricted Upload of File with Dangerous Type",
        "severity": "critical",
        "mitigation": "Validate MIME type and file magic bytes; store files outside webroot with randomized names.",
        "owasp": "A04:2021-Insecure Design",
    },
    "CWE-611": {
        "name": "Improper Restriction of XML External Entity Reference (XXE)",
        "severity": "high",
        "mitigation": "Disable external entity resolution (DTD) in XML parsers.",
        "owasp": "A05:2021-Security Misconfiguration",
    },
    "CWE-306": {
        "name": "Missing Authentication for Critical Function",
        "severity": "critical",
        "mitigation": "Enforce authentication middleware globally on private routes.",
        "owasp": "A07:2021-Identification and Authentication Failures",
    },
}

# OWASP 2021 Top 10 Descriptions
OWASP_2021_TOP10 = {
    "A01:2021-Broken Access Control": "Violations of principle of least privilege, IDOR, path traversal, BOLA.",
    "A02:2021-Cryptographic Failures": "Exposure of sensitive data in transit or rest, broken algorithms, weak keys.",
    "A03:2021-Injection": "SQL, NoSQL, OS command, LDAP, and XSS injection flaws.",
    "A04:2021-Insecure Design": "Missing security architecture controls, rate limiting, and threat modeling.",
    "A05:2021-Security Misconfiguration": "Default credentials, unnecessary open ports, verbose error messages, XXE.",
    "A06:2021-Vulnerable and Outdated Components": "Unpatched CVE dependencies, outdated runtimes, memory vulnerabilities.",
    "A07:2021-Identification and Authentication Failures": "Weak passwords, missing MFA, session hijacking, hardcoded secrets.",
    "A08:2021-Software and Data Integrity Failures": "Insecure deserialization, untrusted CI/CD pipelines, unverified updates.",
    "A09:2021-Security Logging and Monitoring Failures": "Missing audit logs for security transactions, undetected breaches.",
    "A10:2021-Server-Side Request Forgery": "Unchecked fetch of remote resources without IP boundary filtering.",
}

# OWASP API Security Top 10 (2023)
OWASP_API_2023 = {
    "API1:2023-Broken Object Level Authorization": "User accesses another user's object by manipulating IDs in the URI.",
    "API2:2023-Broken Authentication": "Weak token validation, missing session expiration, credential stuffing.",
    "API3:2023-Broken Object Property Level Authorization": "Mass assignment, excessive data exposure of internal model fields.",
    "API4:2023-Unrestricted Resource Consumption": "Lack of execution timeouts, page size limits, or rate limits.",
    "API5:2023-Broken Function Level Authorization": "Regular user accesses administrative endpoints (/api/admin/...).",
    "API6:2023-Unrestricted Access to Sensitive Business Flows": "Automated ticket scalping, excessive SMS verification spam.",
    "API7:2023-Server Side Request Forgery": "API backend fetches client-supplied URIs without network egress filtering.",
    "API8:2023-Security Misconfiguration": "CORS misconfiguration, unhandled debug headers, unpatched API gateways.",
    "API9:2023-Improper Inventory Management": "Zombie APIs, unversioned deprecated staging routes left exposed.",
    "API10:2023-Unsafe Consumption of APIs": "Blind trust in third-party API payloads without validation.",
}


def get_owasp_for_cwe(cwe_id: str) -> Optional[str]:
    """Retrieves standard OWASP category for a given CWE identifier."""
    return CWE_TO_OWASP_MAP.get(cwe_id.upper().strip())


def get_cwe_details(cwe_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves taxonomy metadata for a CWE identifier."""
    return CWE_TAXONOMY.get(cwe_id.upper().strip())


def generate_taxonomy_samples() -> List[SecuritySample]:
    """Generates structured ground-truth samples for CWE classification and OWASP mapping tasks."""
    samples = []
    for cwe_id, details in CWE_TAXONOMY.items():
        owasp_cat = details["owasp"]
        
        # 1. CWE Classification Sample
        s_cwe = SecuritySample(
            sample_id=f"tax-cwe-{cwe_id.lower()}",
            source="mitre_cwe",
            source_id=cwe_id,
            application_archetype="security_taxonomy",
            domain="vulnerability_standards",
            repository="mitre/cwe",
            language="agnostic",
            framework="standard",
            requirement=f"Classify and mitigate {details['name']}.",
            developer_prompt=f"Explain {cwe_id} ({details['name']}) and identify its OWASP alignment and mitigation.",
            project_context={"cwe": cwe_id, "standard": "MITRE CWE Top 25"},
            source_code=f"// Security Definition for {cwe_id}: {details['name']}\n// Mitigation: {details['mitigation']}",
            threat_description=f"Risk exposure resulting from {details['name']}.",
            security_requirements=[details["mitigation"]],
            cwe=[cwe_id],
            cwe_description=details["name"],
            owasp_category=[owasp_cat],
            risk_level=details["severity"],
            severity=details["severity"],
            confidence=ConfidenceLevel.HIGH,
            security_strategy=[f"Enforce standard controls for {details['name']}."],
            capability_plan=["ShiftGuard-AccessAuditor", "ShiftGuard-StaticAnalyzer"],
            vulnerability_explanation=f"{cwe_id} ({details['name']}) represents a high-impact security flaw mapped to {owasp_cat}.",
            task_type=TaskType.CWE_CLASSIFY,
            provenance=ProvenanceMetadata(
                source_name="mitre_cwe_top25",
                source_id=cwe_id,
                repository="mitre/cwe",
                license="CC-BY-4.0",
                is_synthetic=False,
                evidence_source="mitre_official_taxonomy",
                archetype="security_taxonomy",
            ),
            license="CC-BY-4.0",
        )
        samples.append(s_cwe)

        # 2. OWASP Mapping Sample
        s_owasp = SecuritySample(
            sample_id=f"tax-owasp-{cwe_id.lower()}",
            source="owasp_taxonomy",
            source_id=f"{cwe_id}-{owasp_cat[:7]}",
            application_archetype="security_taxonomy",
            domain="vulnerability_standards",
            repository="owasp/top10",
            language="agnostic",
            framework="standard",
            requirement=f"Map {cwe_id} ({details['name']}) to OWASP Top 10 (2021).",
            developer_prompt=f"Determine the primary OWASP Top 10 category for {cwe_id}.",
            project_context={"cwe": cwe_id, "owasp": owasp_cat},
            source_code=f"// Mapping Rule: {cwe_id} -> {owasp_cat}",
            threat_description=f"Standard mapping between {cwe_id} and {owasp_cat}.",
            security_requirements=[f"Adhere to OWASP recommendations under {owasp_cat}."],
            cwe=[cwe_id],
            cwe_description=details["name"],
            owasp_category=[owasp_cat],
            risk_level=details["severity"],
            severity=details["severity"],
            confidence=ConfidenceLevel.HIGH,
            security_strategy=[f"Comply with {owasp_cat} defensive architectural guidelines."],
            capability_plan=["ShiftGuard-TaxonomyMapper"],
            vulnerability_explanation=f"{cwe_id} corresponds to {owasp_cat}: {OWASP_2021_TOP10.get(owasp_cat, '')}",
            task_type=TaskType.OWASP_MAP,
            provenance=ProvenanceMetadata(
                source_name="owasp_top10_2021",
                source_id=f"{cwe_id}-{owasp_cat[:7]}",
                repository="owasp/top10",
                license="CC-BY-SA-4.0",
                is_synthetic=False,
                evidence_source="owasp_canonical_mapping",
                archetype="security_taxonomy",
            ),
            license="CC-BY-SA-4.0",
        )
        samples.append(s_owasp)

    return samples
