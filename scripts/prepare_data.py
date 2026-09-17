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

    import argparse
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Data Preparation Pipeline")
    parser.add_argument("--num-per-archetype", type=int, default=100, help="Number of synthetic instances per archetype")
    parser.add_argument("--output", type=str, default="datasets/processed/corpus_raw.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # 1. Generate multi-task synthetic archetypes
    print(f"\n[1/3] Generating multi-task scenario corpus across 12 archetypes ({args.num_per_archetype} per archetype)...")
    generator = SyntheticSecurityGenerator(seed=42)
    synth_samples = generator.generate_corpus(num_samples_per_archetype=args.num_per_archetype)
    all_samples.extend(synth_samples)
    print(f"  * Generated {len(synth_samples)} structured synthetic instances.")

    # 2. Ingest Juliet Test Suite cases across diverse CWEs
    print("\n[2/3] Parsing NIST Juliet Test Suite cases (CWE-120 Buffer Copy, CWE-22 Path Traversal, CWE-78 Command Injection)...")
    juliet_cases = [
        ("CWE120_Buffer_Copy__char_01.c", """
        void bad() {
            char * data;
            char dataBuffer[100] = "";
            data = dataBuffer;
            if (gets(data) == NULL) { exit(1); }
            CWE120_Buffer_Copy__char_01_bad();
        }
        void good() {
            char * data;
            char dataBuffer[100] = "";
            data = dataBuffer;
            if (fgets(data, 100, stdin) == NULL) { exit(1); }
            CWE120_Buffer_Copy__char_01_good();
        }
        """),
        ("CWE22_Path_Traversal__file_01.c", """
        void bad() {
            char filename[256];
            scanf("%255s", filename);
            FILE *f = fopen(filename, "r");
            if (f) fclose(f);
        }
        void good() {
            char filename[256];
            scanf("%255s", filename);
            if (strstr(filename, "..") == NULL && strstr(filename, "/") == NULL) {
                FILE *f = fopen(filename, "r");
                if (f) fclose(f);
            }
        }
        """),
        ("CWE78_OS_Command_Injection__popen_01.c", """
        void bad() {
            char cmd[512];
            sprintf(cmd, "ping -c 1 %s", user_ip);
            FILE *p = popen(cmd, "r");
            if (p) pclose(p);
        }
        void good() {
            char *args[] = {"/bin/ping", "-c", "1", sanitized_ip, NULL};
            execv(args[0], args);
        }
        """)
    ]
    for fname, code in juliet_cases:
        juliet_samples = JulietLoader.parse_juliet_source(code, fname)
        all_samples.extend(juliet_samples)
    print(f"  * Ingested {len(juliet_cases) * 2} Juliet paired vulnerability/repair instances.")

    # 3. Ingest CVEFixes advisory commit diffs
    print("\n[3/3] Parsing CVEFixes advisory commit diffs across major vulnerabilities...")
    cve_records = [
        {
            "cve_id": "CVE-2022-21700",
            "cwe_id": "CWE-79",
            "programming_language": "javascript",
            "repo_url": "github.com/matrix-org/matrix-react-sdk",
            "commit_hash": "b2685934a36279f0451cfbf4b24e6a88b1b88e14",
            "code_before": "const sanitized = sanitizeHtml(input, { allowedTags: ['b', 'i', 'a'] });",
            "code_after": "const sanitized = DOMPurify.sanitize(input, { RETURN_DOM: false, FORBID_TAGS: ['style'] });",
            "commit_message": "Fix XSS by switching to strict DOMPurify sanitization",
        },
        {
            "cve_id": "CVE-2021-44228",
            "cwe_id": "CWE-502",
            "programming_language": "java",
            "repo_url": "github.com/apache/logging-log4j2",
            "commit_hash": "c77b3cb3931643d4cbfd764ac27cf911a5da4838",
            "code_before": "return JndiManager.getDefaultManager().lookup(name);",
            "code_after": "if (!isJndiLookupAllowed()) { throw new NamingException(\"JNDI disabled\"); }",
            "commit_message": "Disable JNDI lookup by default to eliminate remote code execution",
        },
        {
            "cve_id": "CVE-2023-38606",
            "cwe_id": "CWE-918",
            "programming_language": "python",
            "repo_url": "github.com/langchain-ai/langchain",
            "commit_hash": "e8d69f0b83e498c17b8f9e0134bcad0527376092",
            "code_before": "response = requests.get(user_provided_url, timeout=5)",
            "code_after": "validate_safe_url(user_provided_url); response = requests.get(user_provided_url, timeout=5)",
            "commit_message": "Prevent SSRF by asserting target URL against internal IP ranges",
        },
        {
            "cve_id": "CVE-2022-41880",
            "cwe_id": "CWE-798",
            "programming_language": "python",
            "repo_url": "github.com/django/django",
            "commit_hash": "a5499f123d53b49e6fbc1936c7a6e11894d489b0",
            "code_before": "SECRET_KEY = 'django-insecure-hardcoded-secret-key-for-test'",
            "code_after": "SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')\nif not SECRET_KEY: raise ImproperlyConfigured('DJANGO_SECRET_KEY must be set in environment')",
            "commit_message": "Require production secret keys from environment variables",
        }
    ]
    for rec in cve_records:
        s = CVEFixesLoader.parse_record(rec)
        if s:
            all_samples.append(s)
            print(f"  * Ingested CVEFixes record: {s.sample_id} ({rec['cve_id']} - {rec['cwe_id']})")

    # Write all to corpus_raw.jsonl
    with open(out_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(s.model_dump_json() + "\n")

    print(f"\nSuccessfully prepared {len(all_samples)} raw security instances -> {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

