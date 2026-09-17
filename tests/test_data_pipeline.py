"""Unit Tests for ShiftGuard-SecLM Data Ingestion, Deduplication, and Splitting.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from datasets.schemas.security_sample import (
    SecuritySample,
    TaskType,
    ConfidenceLevel,
    SeverityLevel,
    SecurityGroundTruth,
    ProvenanceMetadata,
)
from src.data.dedup.exact_dedup import normalize_code, compute_code_hash, ExactDeduplicator
from src.data.dedup.minhash_lsh import MinHash, MinHashLSH, get_shingles
from src.data.split.leakage_split import LeakageSafeSplitter
from src.data.loaders.juliet_loader import JulietLoader
from src.data.loaders.taxonomy_loader import get_owasp_for_cwe, get_cwe_details


def test_code_normalization():
    code_a = """
    # This is a comment
    def login(user, password):
        \"\"\"Docstring to strip\"\"\"
        return check(user, password)
    """
    code_b = """
    def login(user, password):
        // JS style or inline
        return check(user, password)
    """
    norm_a = normalize_code(code_a)
    norm_b = normalize_code(code_b)
    assert "Docstring" not in norm_a
    assert "#" not in norm_a
    assert "//" not in norm_b
    assert "return check(user, password)" in norm_a


def test_exact_deduplication():
    dedup = ExactDeduplicator()
    code1 = "def transfer(a, b): return a - b"
    code2 = "def transfer(a, b):\n    # Comment\n    return a - b"
    code3 = "def receive(x): return x"

    assert not dedup.is_duplicate(code=code1)
    # code2 is normalized equivalent to code1
    assert dedup.is_duplicate(code=code2)
    assert not dedup.is_duplicate(code=code3)
    assert dedup.stats()["duplicates_filtered"] == 1


def test_minhash_lsh():
    lsh = MinHashLSH(threshold=0.80, num_perm=64)
    code_base = "def execute_transaction(account_id, amount):\n    cursor.execute('UPDATE accounts SET bal = bal - ' + amount)"
    code_near = "def execute_transaction(acc_id, amount):\n    cursor.execute('UPDATE accounts SET bal = bal - ' + amount)"
    code_distinct = "def calculate_statistics(numbers):\n    return sum(numbers) / len(numbers)"

    assert lsh.insert("base", code_base)
    # code_near has > 80% similarity to code_base
    assert not lsh.insert("near", code_near)
    # code_distinct is completely different
    assert lsh.insert("distinct", code_distinct)


def test_taxonomy_mapping():
    assert get_owasp_for_cwe("CWE-89") == "A03:2021-Injection"
    assert get_owasp_for_cwe("CWE-287") == "A07:2021-Identification and Authentication Failures"
    assert get_cwe_details("CWE-89")["name"] == "SQL Injection"


def test_leakage_safe_split():
    splitter = LeakageSafeSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    samples = []
    # Create 20 samples across 4 distinct repositories
    for repo_idx in range(4):
        for sample_idx in range(5):
            s = SecuritySample(
                sample_id=f"test-repo{repo_idx}-{sample_idx}",
                task=TaskType.THREAT_ANALYSIS,
                requirement="Sample requirement",
                prompt="Sample prompt",
                context={"language": "python"},
                ground_truth=SecurityGroundTruth(threats=["Threat A"]),
                confidence=ConfidenceLevel.HIGH,
                metadata=ProvenanceMetadata(
                    source_name="test",
                    repository=f"org/repo-{repo_idx}",
                ),
            )
            samples.append(s)

    train_set, val_set, test_set = splitter.partition_samples(samples)

    train_repos = {s.metadata.repository for s in train_set}
    val_repos = {s.metadata.repository for s in val_set}
    test_repos = {s.metadata.repository for s in test_set}

    # Verify zero repository leakage between splits
    assert len(train_repos.intersection(val_repos)) == 0, "Leakage between train and validation!"
    assert len(train_repos.intersection(test_repos)) == 0, "Leakage between train and test!"
    assert len(val_repos.intersection(test_repos)) == 0, "Leakage between validation and test!"


if __name__ == "__main__":
    print("Running ShiftGuard-SecLM Data Pipeline Tests...")
    test_code_normalization()
    print("[PASS] Code Normalization passed")
    test_exact_deduplication()
    print("[PASS] Exact Deduplication passed")
    test_minhash_lsh()
    print("[PASS] MinHash LSH Near-Duplicate passed")
    test_taxonomy_mapping()
    print("[PASS] Taxonomy Mapping passed")
    test_leakage_safe_split()
    print("[PASS] Leakage-Safe Split passed (Zero Repository Overlap)")
    print("\nAll Data Ingestion & Deduplication Tests PASSED successfully!")

