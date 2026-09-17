"""Security Taxonomy Loader: MITRE CWE & OWASP Knowledge Base.

Provides structured mapping between CWE vulnerabilities, OWASP categories,
attack patterns, and standard mitigations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any

# Canonical CWE to OWASP 2021 mapping
CWE_TO_OWASP_MAP: Dict[str, str] = {
    "CWE-89": "A03:2021-Injection",
    "CWE-78": "A03:2021-Injection",
    "CWE-79": "A03:2021-Injection",
    "CWE-287": "A07:2021-Identification and Authentication Failures",
    "CWE-384": "A07:2021-Identification and Authentication Failures",
    "CWE-200": "A01:2021-Broken Access Control",
    "CWE-22": "A01:2021-Broken Access Control",
    "CWE-639": "A01:2021-Broken Access Control",
    "CWE-311": "A02:2021-Cryptographic Failures",
    "CWE-327": "A02:2021-Cryptographic Failures",
    "CWE-328": "A02:2021-Cryptographic Failures",
    "CWE-16": "A05:2021-Security Misconfiguration",
    "CWE-770": "A04:2021-Insecure Design",
    "CWE-502": "A08:2021-Software and Data Integrity Failures",
    "CWE-798": "A07:2021-Identification and Authentication Failures",
    "CWE-918": "A10:2021-Server-Side Request Forgery",
    "CWE-119": "A06:2021-Vulnerable and Outdated Components",
    "CWE-787": "A06:2021-Vulnerable and Outdated Components",
    "CWE-416": "A06:2021-Vulnerable and Outdated Components",
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
        "mitigation": "Context-aware HTML escaping and Content Security Policy (CSP) headers.",
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
}


def get_owasp_for_cwe(cwe_id: str) -> Optional[str]:
    """Retrieves standard OWASP category for a given CWE identifier."""
    return CWE_TO_OWASP_MAP.get(cwe_id.upper().strip())


def get_cwe_details(cwe_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves taxonomy metadata for a CWE identifier."""
    return CWE_TAXONOMY.get(cwe_id.upper().strip())
