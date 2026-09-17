# Dataset Plan: ShiftGuard-SecCorpus & ShiftGuard-SecBench

## 1. Overview
The ShiftGuard training corpus is built to support multi-task contextual security reasoning without compromising research integrity. It avoids indiscriminate web crawling, strictly tracks provenance, and eliminates data leakage between training and evaluation splits.

---

## 2. Public Dataset Ingestion & License Compliance

| Dataset | Stated License | Content Description | Ingestion Policy |
| :--- | :--- | :--- | :--- |
| **Juliet Test Suite v1.3** | NIST / Public Domain (CC0) | 64,000+ synthetic C/C++ & Java vulnerability test cases mapped to 118 CWEs | Normalized to uniform JSONL; clean/flawed pairs extracted |
| **CVEFixes** | CC-BY 4.0 / Apache 2.0 | Git commit diffs, CVE/CWE classifications, and commit messages across 17 languages | Extraction of pre-patch (vulnerable) and post-patch (secure) functions |
| **OWASP Benchmark v1.2** | GPL-2.0 | 2,740 Java test cases designed to test SAST precision and recall | **Strictly held out for external evaluation** (never placed in pretraining or training) |
| **MITRE CWE & CAPEC** | Public Domain (MITRE) | Canonical hierarchical vulnerability definitions, mitigations, and attack patterns | Structured taxonomy mapping and knowledge ingestion |
| **NIST NVD Feeds** | Public Domain (US Govt.) | CVE advisories, affected software versions, and CVSS scores | Security context and advisory grounding |
| **DiverseVul & PrimeVul** | Academic Research (Open) | Curated C/C++/Java vulnerability datasets with verified CWE annotations | Sliced for Stage 2 multi-task training |

*Redistribution Policy*: Raw third-party data is not checked into git. Reproducible Python scripts with SHA-256 checksums automate download and preprocessing.

---

## 3. Custom Datasets

### 3.1 `ShiftGuard-SecCorpus` (~1.5M instances)
Designed to represent the end-to-end security lifecycle:
```json
{
  "sample_id": "sg-corpus-0012948",
  "requirement": "Implement a user authentication endpoint with rate limiting.",
  "prompt": "Write a FastAPI route /login verifying user password against PostgreSQL database.",
  "context": {
    "language": "python",
    "framework": "fastapi",
    "database": "postgresql",
    "has_auth": true,
    "has_crypto": true
  },
  "code": "...",
  "security_findings": [
    {
      "source": "semgrep",
      "rule_id": "python.fastapi.security.audit.raw-sql",
      "cwe": "CWE-89",
      "severity": "high",
      "line": 42
    }
  ],
  "ground_truth": {
    "threats": ["SQL Injection", "Credential Stuffing", "Timing Attacks"],
    "cwe": ["CWE-89", "CWE-307", "CWE-208"],
    "owasp": ["A03:2021-Injection", "A07:2021-Identification and Authentication Failures"],
    "risk": {
      "severity": "critical",
      "score": 9.2,
      "confidence": 0.95
    },
    "security_requirements": [
      "Use parameterized queries or SQLAlchemy ORM.",
      "Implement slow-hash Argon2id password verification.",
      "Add Redis-backed sliding window rate limiter (max 5 attempts/min)."
    ],
    "agent_plan": [
      {"agent": "auth_agent", "action": "audit_token_generation"},
      {"agent": "injection_agent", "action": "verify_parameterization"}
    ],
    "repair": "...",
    "verification_status": "fixed"
  },
  "confidence_level": "high"
}
```

### 3.2 `ShiftGuard-SecBench` (1,500 held-out scenarios)
- Sourced across 12 distinct software application archetypes (Banking, Healthcare, E-commerce, SaaS Admin, IoT API, OAuth Provider, File Storage, Telemetry Backend, Chat/Messaging, Streaming Service, CI/CD Webhook, Payment Gateway).
- Strictly isolated: Zero overlap in repository source, application logic, or author provenance with the training set.

---

## 4. Leakage Prevention Protocol

1. **Exact Deduplication**: SHA-256 hashes of normalized code (comments and whitespace stripped).
2. **Near-Duplicate Detection**: MinHash + Locality Sensitive Hashing (LSH) with Jaccard threshold $> 0.80$ to eliminate prompt and code variants.
3. **Multi-Level Split Partitioning**:
   - Split by **Repository / Project** (no repository appears in more than one split).
   - Split by **Application Archetype**.
   - Temporal split on CVEs (pre-2023 for training, post-2023 for zero-shot testing).

