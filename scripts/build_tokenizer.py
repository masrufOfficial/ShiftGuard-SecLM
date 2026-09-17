#!/usr/bin/env python3
"""Build and benchmark the ShiftGuard-SecLM domain-specialized tokenizer."""

import sys
from pathlib import Path

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.train_tokenizer import (
    train_security_tokenizer,
    test_tokenizer_efficiency,
    ALL_SPECIAL_TOKENS,
)

SAMPLE_CORPUS = """
# Python Security Example
import sqlite3
import hashlib
from fastapi import FastAPI, HTTPException, Depends

app = FastAPI(title="Secure Banking API")

def authenticate_user(db: sqlite3.Connection, username: str, password_hash: str):
    # Parameterized SQL query preventing CWE-89 (SQL Injection)
    cursor = db.cursor()
    cursor.execute("SELECT id, role FROM users WHERE username = ? AND password = ?", (username, password_hash))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication failed")
    return {"id": user[0], "role": user[1]}

// JavaScript / Express Example
const express = require('express');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');

const app = express();
app.use(helmet()); // OWASP A05:2021 Security Misconfiguration defense

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100, // CWE-770: Allocation of Resources Without Limits or Throttling
    standardHeaders: true,
    legacyHeaders: false,
});
app.use('/api/', limiter);

// Security Evidence JSON
{
  "finding_id": "SEC-FINDING-001",
  "tool": "semgrep",
  "rule": "python.lang.security.audit.sqli.psycopg2",
  "cwe": "CWE-89",
  "owasp": "A03:2021-Injection",
  "severity": "HIGH",
  "file": "app/services/billing.py",
  "line": 104,
  "confidence": 0.94,
  "recommendation": "Use parameterized queries or SQLAlchemy ORM sessions instead of raw string formatting."
}

CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
The software constructs all or part of an SQL command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command when it is sent to a downstream component.

OWASP API Security Top 10:
API1:2023 Broken Object Level Authorization (BOLA)
API2:2023 Broken Authentication
API3:2023 Broken Object Property Level Authorization
API4:2023 Unrestricted Resource Consumption
API5:2023 Broken Function Level Authorization
API6:2023 Unrestricted Access to Sensitive Business Flows
API7:2023 Server Side Request Forgery
API8:2023 Security Misconfiguration
API9:2023 Improper Inventory Management
API10:2023 Unsafe Consumption of APIs
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Tokenizer Builder")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Maximum vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=1, help="Minimum token frequency")
    parser.add_argument("--output-dir", type=str, default="datasets/processed/tokenizer", help="Output directory")
    args = parser.parse_args()

    corpus_dir = Path("datasets/samples")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = corpus_dir / "tokenizer_complete_corpus.txt"

    # Aggregate text from corpus_raw.jsonl and sample corpus
    print("[1/3] Aggregating security text, code snippets, and vulnerability taxonomies...")
    text_blocks = [SAMPLE_CORPUS * 10]

    raw_jsonl = Path("datasets/processed/corpus_raw.jsonl")
    if raw_jsonl.exists():
        import json
        with open(raw_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if "prompt" in data and data["prompt"]:
                    text_blocks.append(data["prompt"])
                if "code" in data and data["code"]:
                    text_blocks.append(data["code"])
                gt = data.get("ground_truth", {})
                for threat in gt.get("threats", []):
                    text_blocks.append(str(threat))
                for req in gt.get("security_requirements", []):
                    text_blocks.append(str(req))
                for rep in gt.get("repair_guidance", []):
                    text_blocks.append(str(rep))
                for cwe in gt.get("cwe_ids", []):
                    text_blocks.append(str(cwe))
                for owasp in gt.get("owasp_categories", []):
                    text_blocks.append(str(owasp))

    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(text_blocks))

    print(f"  * Aggregated corpus written to {corpus_file} ({corpus_file.stat().st_size:,} bytes).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/3] Training Byte-Level BPE tokenizer (Target Vocab: {args.vocab_size:,}, Min Freq: {args.min_frequency})...")
    tokenizer = train_security_tokenizer(
        training_files=[str(corpus_file)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=output_dir,
    )

    print(f"  * Tokenizer successfully trained! Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"  * Special tokens registered: {len(ALL_SPECIAL_TOKENS)}")

    # Benchmark efficiency on representative test inputs
    print("\n[3/3] Evaluating tokenizer compression & zero-OOV handling...")
    test_cases = {
        "Python SQLi": "cursor.execute('SELECT * FROM accounts WHERE id = ' + user_input)",
        "CWE Definition": "CWE-89: Improper Neutralization of Special Elements used in an SQL Command",
        "OWASP Taxonomy": "OWASP A03:2021-Injection and API1:2023 Broken Object Level Authorization",
        "Special Tokens": "<SEC_CONTEXT> <LANG:python> <TASK:THREAT_ANALYSIS> <SCHEMA_START>",
        "Hex/Bytecode Shellcode": "\\x31\\xc0\\x50\\x68\\x2f\\x2f\\x73\\x68\\x68\\x2f\\x62\\x69\\x6e",
    }

    results = test_tokenizer_efficiency(tokenizer, test_cases)
    print("\n--- Tokenizer Compression & Representation Evaluation ---")
    for category, stats in results.items():
        print(f"> {category}:")
        print(f"    Chars: {stats['num_chars']} | Tokens: {stats['num_tokens']} | Bytes/Token: {stats['bytes_per_token']}")
        print(f"    Preview: {stats['tokens_preview']}")

    print(f"\nTokenizer artifacts successfully saved to {output_dir}/")


if __name__ == "__main__":
    main()

