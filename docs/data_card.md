# Dataset Card: ShiftGuard-SecCorpus & ShiftGuard-SecBench

## 1. Dataset Summary
- **Name**: ShiftGuard-SecCorpus & ShiftGuard-SecBench
- **Purpose**: Domain-specialized security pretraining and multi-task evaluation for Shift-Left security reasoning.
- **Languages Covered**: Python, JavaScript, TypeScript, Java, C/C++, Go, SQL.

## 2. Ingestion & Provenance
- **Public Sources**: Juliet Test Suite v1.3 (NIST, CC0), CVEFixes (CC-BY 4.0), MITRE CWE/CAPEC (Public Domain), NIST NVD (Public Domain), DiverseVul.
- **Evaluation Benchmark**: OWASP Benchmark v1.2 (GPL-2.0, evaluation only) and custom 1,500 held-out scenarios (`ShiftGuard-SecBench`).
- **Deduplication**: Exact hash deduplication + MinHash/LSH near-duplicate pruning.
- **Leakage Prevention**: Repository-level and application-archetype partition splits.

