#!/usr/bin/env python3
"""Build and benchmark the ShiftGuard-SecLM domain-specialized tokenizer.

STRICT RESEARCH INTEGRITY:
Trains exclusively on the training partition (`datasets/manifests/train.jsonl`
or specified `--train-data`). Quarantined validation and test partitions are
strictly excluded to guarantee zero test leakage into vocabulary construction.
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.train_tokenizer import (
    train_security_tokenizer,
    test_tokenizer_efficiency,
    ALL_SPECIAL_TOKENS,
)
from src.data.loaders.taxonomy_loader import (
    CWE_TAXONOMY,
    OWASP_2021_TOP10,
    OWASP_API_2023,
    CWE_TO_OWASP_MAP,
)


SAMPLE_CORPUS_SEED = """
# ShiftGuard-SecLM Domain Vocabulary Seed
import sqlite3
import hashlib
import hmac
import secrets
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

app = FastAPI(title="Secure Banking Gateway")

def authenticate_user(db: sqlite3.Connection, username: str, password_hash: str):
    # Parameterized SQL query preventing CWE-89 (SQL Injection)
    cursor = db.cursor()
    cursor.execute("SELECT id, role FROM users WHERE username = ? AND password = ?", (username, password_hash))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication failed")
    return {"id": user[0], "role": user[1]}

const express = require('express');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const DOMPurify = require('dompurify');

const app = express();
app.use(helmet());

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100,
    standardHeaders: true,
    legacyHeaders: false,
});
app.use('/api/', limiter);

// Security Evidence Finding
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
"""


def extract_text_from_train_record(record: Dict[str, Any]) -> List[str]:
    """Extracts all text fields from a single training record for tokenizer corpus building."""
    blocks: List[str] = []
    
    # Prompt & requirements
    prompt = record.get("developer_prompt") or record.get("prompt")
    if prompt:
        blocks.append(str(prompt))
    req = record.get("requirement")
    if req:
        blocks.append(str(req))
        
    # Code blocks
    code = record.get("source_code") or record.get("code")
    if code:
        blocks.append(str(code))
    repair = record.get("secure_repair") or record.get("ground_truth", {}).get("repair_code")
    if repair:
        blocks.append(str(repair))
        
    # Explanations and threats
    threat = record.get("threat_description")
    if threat:
        blocks.append(str(threat))
    expl = record.get("vulnerability_explanation") or record.get("ground_truth", {}).get("explanation")
    if expl:
        blocks.append(str(expl))
        
    # Requirements and strategies
    reqs = record.get("security_requirements") or record.get("ground_truth", {}).get("security_requirements", [])
    if isinstance(reqs, list):
        for r in reqs:
            blocks.append(str(r))
    elif isinstance(reqs, str):
        blocks.append(reqs)
        
    strats = record.get("security_strategy") or record.get("ground_truth", {}).get("security_strategy", [])
    if isinstance(strats, list):
        for s in strats:
            blocks.append(str(s))
    elif isinstance(strats, str):
        blocks.append(strats)
        
    # Taxonomies
    cwes = record.get("cwe") or record.get("ground_truth", {}).get("cwe", [])
    if isinstance(cwes, list):
        for c in cwes:
            blocks.append(str(c))
    elif isinstance(cwes, str):
        blocks.append(cwes)
        
    owasps = record.get("owasp_category") or record.get("ground_truth", {}).get("owasp", [])
    if isinstance(owasps, list):
        for o in owasps:
            blocks.append(str(o))
    elif isinstance(owasps, str):
        blocks.append(owasps)
        
    # Findings
    findings = record.get("security_findings") or []
    if isinstance(findings, list):
        for f in findings:
            if isinstance(f, dict):
                msg = f.get("message")
                if msg:
                    blocks.append(str(msg))
                    
    return blocks


def main():
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Tokenizer Builder")
    parser.add_argument(
        "--train-data",
        type=str,
        default="datasets/manifests/train.jsonl",
        help="Path to training data partition (ZERO leakage allowed from val/test)",
    )
    parser.add_argument(
        "--corpus-output",
        type=str,
        default="datasets/processed/tokenizer_corpus.txt",
        help="Path to save aggregated plain text tokenizer training corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/processed/tokenizer",
        help="Output directory for tokenizer artifacts",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target maximum vocabulary size (Master Spec specifies 32,000)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency in corpus to be included in BPE vocabulary",
    )
    args = parser.parse_args()

    train_path = Path(args.train_data)
    if not train_path.exists():
        # Fallback to datasets/splits/train.jsonl if manifests not yet populated
        alt_path = Path("datasets/splits/train.jsonl")
        if alt_path.exists():
            train_path = alt_path
        else:
            raise FileNotFoundError(f"Training data file not found at {train_path} or {alt_path}")

    corpus_out_path = Path(args.corpus_output)
    corpus_out_path.parent.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" SHIFTGUARD-SECLM: DOMAIN-SPECIALIZED BPE TOKENIZER BUILDER")
    print("=" * 80)
    print(f"Training Partition Source: {train_path}")
    print(f"Tokenization Output Dir:   {output_dir}")
    print(f"Corpus Text Export:        {corpus_out_path}")
    print(f"Target Vocabulary Size:    {args.vocab_size:,}")
    print(f"Minimum Frequency:         {args.min_frequency}")
    print("-" * 80)

    # 1. Aggregate text strictly from training partition
    print("\n[1/4] Aggregating domain security training corpus (Zero Leakage Enforcement)...")
    text_corpus_blocks: List[str] = [SAMPLE_CORPUS_SEED.strip()]

    # Add canonical taxonomy descriptions
    for cwe_id, meta in CWE_TAXONOMY.items():
        text_corpus_blocks.append(f"{cwe_id}: {meta['name']}. {meta['mitigation']}")
    for cat_id, desc in OWASP_2021_TOP10.items():
        text_corpus_blocks.append(f"{cat_id}: {desc}")
    for cat_id, desc in OWASP_API_2023.items():
        text_corpus_blocks.append(f"{cat_id}: {desc}")

    sample_count = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            # Enforce that validation/test records are strictly absent
            split_tag = rec.get("split")
            if split_tag in ("val", "validation", "test"):
                raise ValueError(
                    f"CRITICAL LEAKAGE ERROR: Found record with split='{split_tag}' inside training corpus!"
                )
            extracted = extract_text_from_train_record(rec)
            text_corpus_blocks.extend(extracted)
            sample_count += 1

    # Write aggregated corpus to disk
    corpus_text = "\n\n".join(text_corpus_blocks)
    with open(corpus_out_path, "w", encoding="utf-8") as f:
        f.write(corpus_text)

    corpus_size_bytes = corpus_out_path.stat().st_size
    print(f"  * Aggregated text from {sample_count:,} training samples + canonical taxonomies.")
    print(f"  * Total corpus size: {corpus_size_bytes:,} bytes ({corpus_size_bytes / (1024 * 1024):.2f} MB).")
    print(f"  * Corpus saved to: {corpus_out_path}")

    # 2. Train Byte-Level BPE tokenizer
    print(f"\n[2/4] Training Byte-Level BPE Tokenizer (Vocab: {args.vocab_size:,}, Min Freq: {args.min_frequency})...")
    tokenizer = train_security_tokenizer(
        training_files=[str(corpus_out_path)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=output_dir,
    )

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"  * Tokenizer trained successfully! Actual Vocabulary Size: {actual_vocab_size:,}")
    print(f"  * Control and Special Tokens Registered: {len(ALL_SPECIAL_TOKENS)}")

    # 3. Benchmark efficiency and compression
    print("\n[3/4] Evaluating tokenization compression and domain representation...")
    test_cases = {
        "Python Parameterized SQL": "cursor.execute('SELECT id, role FROM users WHERE username = ?', (username,))",
        "Python Vulnerable Raw SQL": "cursor.execute('SELECT * FROM users WHERE id = ' + user_input)",
        "CWE Taxonomy String": "CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')",
        "OWASP Taxonomy String": "A03:2021-Injection and API1:2023 Broken Object Level Authorization (BOLA)",
        "Special Boundary Tokens": "<SEC_CONTEXT> <LANG:python> <FRAMEWORK:fastapi> <TASK:THREAT_ANALYSIS> <SCHEMA_START>",
        "JavaScript DOMPurify Sanitization": "DOMPurify.sanitize(userInput, { RETURN_DOM: false, FORBID_TAGS: ['script'] })",
        "C Buffer Overflow & Bound Guard": "if (len >= sizeof(dest_buf)) { return -1; } memcpy(dest_buf, src, len);",
        "Hex/Bytecode Shellcode": "\\x31\\xc0\\x50\\x68\\x2f\\x2f\\x73\\x68\\x68\\x2f\\x62\\x69\\x6e",
        "JSON Evidence Finding": '{"rule": "sec.audit.sqli", "severity": "HIGH", "confidence": 0.95}',
    }

    efficiency_results = test_tokenizer_efficiency(tokenizer, test_cases)
    total_chars = sum(s["num_chars"] for s in efficiency_results.values())
    total_tokens = sum(s["num_tokens"] for s in efficiency_results.values())
    total_bytes = sum(s["num_bytes"] for s in efficiency_results.values())
    avg_bytes_per_token = round(total_bytes / max(total_tokens, 1), 3)

    print("\n--- Tokenizer Compression Benchmark ---")
    for category, stats in efficiency_results.items():
        print(f"  > {category:35s}: {stats['num_tokens']:3d} tokens | {stats['bytes_per_token']:4.2f} bytes/tok | Preview: {stats['tokens_preview'][:6]}")
    print(f"  * Overall Average Compression Ratio: {avg_bytes_per_token:.2f} bytes/token across security modalities.")

    # 4. Domain Security Vocabulary Inspection
    print("\n[4/4] Inspecting Security Domain Token Representation...")
    security_terms = [
        "CWE", "OWASP", "vulnerability", "injection", "buffer", "overflow",
        "sanitize", "authentication", "authorization", "deserialize", "payload",
        "tampering", "privilege", "escalation", "exploit", "remediation", "mitigation"
    ]
    vocab = tokenizer.get_vocab()
    term_status = {}
    for term in security_terms:
        # Check if term or Byte-prefixed term exists in vocabulary
        in_vocab = term in vocab or f"Ġ{term}" in vocab or term.lower() in vocab or f"Ġ{term.lower()}" in vocab
        tokens = tokenizer.encode(term).tokens
        term_status[term] = {
            "in_vocabulary": in_vocab,
            "subword_split": tokens,
            "token_count": len(tokens),
        }
        print(f"  * Term '{term}': {len(tokens)} token(s) -> {tokens}")

    # Calculate average tokens per sample across training set
    print("\nCalculating sample token distribution across training partition...")
    sample_token_counts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sample_text = " ".join(extract_text_from_train_record(rec))
            tok_len = len(tokenizer.encode(sample_text).ids)
            sample_token_counts.append(tok_len)

    avg_tokens_per_sample = round(sum(sample_token_counts) / max(len(sample_token_counts), 1), 1)
    min_tokens_per_sample = min(sample_token_counts) if sample_token_counts else 0
    max_tokens_per_sample = max(sample_token_counts) if sample_token_counts else 0
    total_tokens_train = sum(sample_token_counts)

    print(f"  * Evaluated {len(sample_token_counts):,} training samples:")
    print(f"    - Total Tokens:            {total_tokens_train:,}")
    print(f"    - Mean Tokens / Sample:    {avg_tokens_per_sample:.1f}")
    print(f"    - Min Tokens / Sample:     {min_tokens_per_sample}")
    print(f"    - Max Tokens / Sample:     {max_tokens_per_sample}")

    # Build report dict
    report = {
        "pipeline": "ShiftGuard-SecLM Tokenizer Build",
        "author": "Masruf Rahman",
        "vocab_size_configured": args.vocab_size,
        "vocab_size_actual": actual_vocab_size,
        "special_tokens_count": len(ALL_SPECIAL_TOKENS),
        "training_partition_path": str(train_path),
        "corpus_bytes": corpus_size_bytes,
        "total_train_samples": len(sample_token_counts),
        "total_train_tokens": total_tokens_train,
        "mean_tokens_per_sample": avg_tokens_per_sample,
        "min_tokens_per_sample": min_tokens_per_sample,
        "max_tokens_per_sample": max_tokens_per_sample,
        "avg_bytes_per_token": avg_bytes_per_token,
        "compression_benchmarks": efficiency_results,
        "security_term_representation": term_status,
        "vocabulary_recommendation_for_1b": {
            "recommended_vocab_size": 32000,
            "rationale": (
                "For a ~1B parameter decoder-only model (d_model=2048, layers=24), a 32,000 vocabulary strikes an "
                "optimal parameter budget balance: embedding parameters = 32,000 * 2,048 = 65,536,000 (~65.5M parameters, "
                "or 6.5% of total model capacity). An excessively large vocabulary (e.g., 64,000 or 100,000) would consume "
                ">130M parameters solely on the token embedding table and final lm_head un-embedding projection without "
                "adding contextual depth, diluting gradients across sparse tokens. Byte-level BPE guarantees 0% OOV for "
                "arbitrary hex payloads, binary shellcode, and novel syntax, while 32k enables compact subword representations "
                "for standard code identifiers and specialized security taxonomies."
            ),
        },
    }

    report_dir = Path("datasets/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "tokenizer_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Also save in datasets/statistics/
    stats_dir = Path("datasets/statistics")
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / "tokenizer_stats.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport and statistics successfully saved to:")
    print(f"  * {report_path}")
    print(f"  * {stats_dir / 'tokenizer_stats.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
