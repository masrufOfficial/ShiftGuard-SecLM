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
    corpus_dir = Path("datasets/samples")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = corpus_dir / "tokenizer_prototype_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write(SAMPLE_CORPUS * 50)  # Replicate to provide sufficient token frequency

    output_dir = Path("datasets/processed/tokenizer")
    print(f"Building tokenizer on {corpus_file}...")
    tokenizer = train_security_tokenizer(
        training_files=[str(corpus_file)],
        vocab_size=1000,  # Small prototype vocabulary for smoke testing
        min_frequency=2,
        output_dir=output_dir,
    )

    print(f"Tokenizer trained! Total vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens registered: {len(ALL_SPECIAL_TOKENS)}")

    # Benchmark efficiency on representative test inputs
    test_cases = {
        "Python SQLi": "cursor.execute('SELECT * FROM accounts WHERE id = ' + user_input)",
        "CWE Definition": "CWE-89: Improper Neutralization of Special Elements used in an SQL Command",
        "OWASP Taxonomy": "OWASP A03:2021-Injection and API1:2023 Broken Object Level Authorization",
        "Special Tokens": "<SEC_CONTEXT> <LANG:python> <TASK:THREAT_ANALYSIS> <SCHEMA_START>",
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

