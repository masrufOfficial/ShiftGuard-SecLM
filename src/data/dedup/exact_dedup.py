"""Exact and Normalized Code Hash Deduplication for ShiftGuard-SecLM.

Normalizes code (removes comments, docstrings, and redundant whitespace)
to generate deterministic SHA-256 fingerprint hashes.
"""

from __future__ import annotations
import re
import hashlib
from typing import Set, Dict, Any, List, Tuple


def normalize_code(code: str, language: str = "python") -> str:
    """Normalizes code to eliminate superficial variations in whitespace and comments."""
    if not code:
        return ""
    
    # 1. Remove single-line comments (# in Python/SQL/YAML, // in JS/TS/Java/C/Go)
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
    
    # 2. Remove multi-line comments (/* ... */)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    
    # 3. For Python, remove triple-quote docstrings
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    
    # 4. Normalize all whitespace sequences (including newlines) to a single space
    normalized = re.sub(r"\s+", " ", code).strip()
    return normalized


def compute_code_hash(code: str, language: str = "python") -> str:
    """Computes SHA-256 hash of normalized code."""
    normalized = normalize_code(code, language)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ExactDeduplicator:
    """Tracks observed code and prompt hashes to eliminate duplicates in dataset streams."""

    def __init__(self):
        self.seen_code_hashes: Set[str] = set()
        self.seen_prompt_hashes: Set[str] = set()
        self.duplicates_filtered: int = 0

    def is_duplicate(self, code: Optional[str] = None, prompt: Optional[str] = None, language: str = "python") -> bool:
        """Returns True if either normalized code or prompt has already been encountered."""
        is_dup = False

        if code:
            chash = compute_code_hash(code, language)
            if chash in self.seen_code_hashes:
                is_dup = True
            else:
                self.seen_code_hashes.add(chash)

        if prompt and not is_dup:
            phash = hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()
            if phash in self.seen_prompt_hashes:
                is_dup = True
            else:
                self.seen_prompt_hashes.add(phash)

        if is_dup:
            self.duplicates_filtered += 1

        return is_dup

    def stats(self) -> Dict[str, int]:
        return {
            "unique_code_count": len(self.seen_code_hashes),
            "unique_prompt_count": len(self.seen_prompt_hashes),
            "duplicates_filtered": self.duplicates_filtered,
        }
