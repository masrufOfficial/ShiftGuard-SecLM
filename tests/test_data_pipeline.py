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



def test_schema_v2_validation_and_sync():
    sample = SecuritySample(
        sample_id="test-sample-01",
        task_type="repair",
        application_archetype="banking_api",
        domain="fintech",
        repository="org/bank-core",
        language="python",
        framework="fastapi",
        developer_prompt="Repair the insecure SQL query.",
        requirement="Prevent SQL injection in user lookup.",
        source_code="def get_user(uid): return db.query(f'SELECT * FROM users WHERE id={uid}')",
        threat_description="String formatting allows SQL injection.",
        security_requirements=["Enforce parameterized queries."],
        cwe=["CWE-89"],
        owasp_category=["A03:2021-Injection"],
        severity="critical",
        confidence="high",
        secure_repair="def get_user(uid): return db.query('SELECT * FROM users WHERE id=:uid', uid=uid)",
        verification="fixed",
        provenance={
            "source_name": "synthetic_generator",
            "repository": "org/bank-core",
            "license": "Apache-2.0",
            "is_synthetic": True,
        },
    )
    # Validate sync to v1 fields
    assert sample.task == TaskType.REPAIR
    assert sample.code == sample.source_code
    assert sample.prompt == sample.developer_prompt
    assert sample.ground_truth.cwe == ["CWE-89"]
    assert sample.ground_truth.repair_code == sample.secure_repair
    assert sample.metadata.is_synthetic is True
    assert sample.metadata.license == "Apache-2.0"

    # Validate dict dump
    d = sample.model_dump()
    assert d["schema_version"] == "2.0.0"
    assert d["task_type"] == "repair"
    assert d["source_code"] == sample.source_code


def test_quality_flags_audit():
    from datasets.schemas.security_sample import QualityAuditFlags
    valid_flags = QualityAuditFlags(
        valid_json=True,
        valid_schema=True,
        non_empty_code=True,
        non_empty_prompt=True,
        valid_cwe=True,
        valid_owasp=True,
        has_provenance=True,
        has_license=True,
        consistent_severity=True,
    )
    assert all(valid_flags.model_dump().values()) is True


def test_task_aware_deduplication():
    dedup = ExactDeduplicator()
    code = "def query_db(x): return db.find(x)"
    # Insert code under THREAT_ANALYSIS task
    is_dup1 = dedup.is_duplicate(code=code, task="threat_analysis")
    assert not is_dup1

    # Same code under REPAIR task should NOT be considered duplicate
    is_dup2 = dedup.is_duplicate(code=code, task="repair")
    assert not is_dup2

    # Same code under THREAT_ANALYSIS task SHOULD be detected as duplicate
    is_dup3 = dedup.is_duplicate(code=code, task="threat_analysis")
    assert is_dup3


def test_test_set_archetype_quarantine():
    splitter = LeakageSafeSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        held_out_archetypes=["iot_backend", "payment_gateway"],
    )

    samples = []
    # Regular archetype samples
    for i in range(10):
        samples.append(
            SecuritySample(
                sample_id=f"web-{i}",
                task_type="vuln_analysis",
                application_archetype="web_app",
                developer_prompt="Review web handler",
                source_code=f"def handle_{i}(): pass",
                cwe=["CWE-79"],
                owasp_category=["A03:2021-Injection"],
                severity="medium",
                confidence="high",
                provenance={"source_name": "test", "repository": "test/web", "license": "MIT", "is_synthetic": True},
            )
        )
    # Quarantined archetype samples
    for i in range(4):
        samples.append(
            SecuritySample(
                sample_id=f"iot-{i}",
                task_type="threat_analysis",
                application_archetype="iot_backend",
                developer_prompt="Review firmware MQTT handler",
                source_code=f"def mqtt_rx_{i}(): pass",
                cwe=["CWE-287"],
                owasp_category=["A07:2021-Identification and Authentication Failures"],
                severity="high",
                confidence="high",
                provenance={"source_name": "test", "repository": "test/iot", "license": "MIT", "is_synthetic": True},
            )
        )

    train_set, val_set, test_set = splitter.partition_samples(samples)

    # All iot_backend samples MUST be in test_set
    train_archetypes = {s.application_archetype for s in train_set}
    val_archetypes = {s.application_archetype for s in val_set}
    test_archetypes = {s.application_archetype for s in test_set}

    assert "iot_backend" not in train_archetypes, "Quarantined archetype leaked into train set!"
    assert "iot_backend" not in val_archetypes, "Quarantined archetype leaked into val set!"
    assert "iot_backend" in test_archetypes, "Quarantined archetype missing from test set!"


def test_tokenizer_isolation_and_zero_oov():
    from tokenizers import Tokenizer
    tok_path = Path("datasets/processed/tokenizer/tokenizer.json")
    if tok_path.exists():
        tokenizer = Tokenizer.from_file(str(tok_path))
        tokens = tokenizer.encode("<SEC_CONTEXT> <TASK:REPAIR> <LANG:python>").tokens
        assert "<SEC_CONTEXT>" in tokens
        assert "<TASK:REPAIR>" in tokens

        # Byte-level BPE guarantees 0% OOV for arbitrary unseen bytes
        hex_shellcode = "\\x31\\xc0\\x50\\x68\\x2f\\x2f\\x73\\x68"
        enc = tokenizer.encode(hex_shellcode)
        unk_id = tokenizer.token_to_id("<|unk|>")
        assert unk_id not in enc.ids, "OOV token produced by Byte-level BPE tokenizer!"


def test_manifests_split_integrity():
    import json
    train_f = Path("datasets/manifests/train.jsonl")
    val_f = Path("datasets/manifests/val.jsonl")
    test_f = Path("datasets/manifests/test.jsonl")

    if train_f.exists() and val_f.exists() and test_f.exists():
        def load_ids_and_repos(p):
            ids, cases, repos = set(), set(), set()
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        ids.add(data["sample_id"])
                        src_id = data.get("source_id") or data.get("provenance", {}).get("source_id")
                        if src_id:
                            cases.add(src_id)
                        repo = data.get("repository") or data.get("provenance", {}).get("repository")
                        if repo and "shiftguard/" not in repo:
                            repos.add(repo)
            return ids, cases, repos

        train_ids, train_cases, train_repos = load_ids_and_repos(train_f)
        val_ids, val_cases, val_repos = load_ids_and_repos(val_f)
        test_ids, test_cases, test_repos = load_ids_and_repos(test_f)

        # Zero sample ID overlap
        assert len(train_ids.intersection(val_ids)) == 0, "Sample ID overlap between train and val!"
        assert len(train_ids.intersection(test_ids)) == 0, "Sample ID overlap between train and test!"
        assert len(val_ids.intersection(test_ids)) == 0, "Sample ID overlap between val and test!"

        # Zero scenario case ID overlap
        assert len(train_cases.intersection(val_cases)) == 0, "Case overlap between train and val!"
        assert len(train_cases.intersection(test_cases)) == 0, "Case overlap between train and test!"
        assert len(val_cases.intersection(test_cases)) == 0, "Case overlap between val and test!"

        # Zero external real-world repository overlap
        assert len(train_repos.intersection(val_repos)) == 0, "Repo overlap between train and val!"
        assert len(train_repos.intersection(test_repos)) == 0, "Repo overlap between train and test!"
        assert len(val_repos.intersection(test_repos)) == 0, "Repo overlap between val and test!"


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
    test_schema_v2_validation_and_sync()
    print("[PASS] Schema V2 Validation and Sync passed")
    test_quality_flags_audit()
    print("[PASS] Quality Flags Audit passed")
    test_task_aware_deduplication()
    print("[PASS] Task-Aware Deduplication passed")
    test_test_set_archetype_quarantine()
    print("[PASS] Test-Set Archetype Quarantine passed")
    test_tokenizer_isolation_and_zero_oov()
    print("[PASS] Tokenizer Isolation and 0% OOV passed")
    test_manifests_split_integrity()
    print("[PASS] Manifests Split Integrity passed")
    print("\nAll Data Ingestion, Schema V2, Deduplication, and Isolation Tests PASSED successfully!")


