"""MinHash and Locality-Sensitive Hashing (LSH) for ShiftGuard-SecLM.

Identifies and prunes near-duplicate code files, vulnerability variations,
and rephrased prompts using Jaccard similarity estimation.
"""

from __future__ import annotations
import re
import hashlib
import random
from typing import Set, List, Dict, Tuple, Optional


def get_shingles(text: str, k: int = 5) -> Set[str]:
    """Generates k-shingles (character n-grams) from normalized text."""
    # Normalize whitespace
    clean_text = re.sub(r"\s+", " ", text.strip().lower())
    if len(clean_text) < k:
        return {clean_text}
    return {clean_text[i : i + k] for i in range(len(clean_text) - k + 1)}


class MinHash:
    """MinHash signature generator using universal hashing."""

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        self.seed = seed
        self.prime = 4294967311  # 2^32 - 5

        # Precompute random linear hash coefficients: h_i(x) = (a * x + b) % prime
        rng = random.Random(seed)
        self.a = [rng.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.prime - 1) for _ in range(num_perm)]

    def compute_signature(self, shingles: Set[str]) -> List[int]:
        """Computes MinHash signature vector of length num_perm."""
        signature = [float("inf")] * self.num_perm

        for shingle in shingles:
            # 32-bit hash value for shingle
            h_val = int(hashlib.md5(shingle.encode("utf-8")).hexdigest()[:8], 16)
            for i in range(self.num_perm):
                perm_val = (self.a[i] * h_val + self.b[i]) % self.prime
                if perm_val < signature[i]:
                    signature[i] = perm_val

        return signature

    @staticmethod
    def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
        """Estimates Jaccard similarity from two MinHash signatures."""
        assert len(sig_a) == len(sig_b)
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)


class MinHashLSH:
    """Locality-Sensitive Hashing table for indexing MinHash signatures.
    
    Partitions signatures into `b` bands of `r` rows.
    Theoretical threshold: t ~= (1/b)^(1/r).
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.minhash = MinHash(num_perm=num_perm)

        # Optimize number of bands (b) and rows (r) such that b * r = num_perm
        best_b, best_r = 16, 8
        min_diff = float("inf")
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                approx_thresh = (1.0 / b) ** (1.0 / r)
                diff = abs(approx_thresh - threshold)
                if diff < min_diff:
                    min_diff = diff
                    best_b, best_r = b, r

        self.b = best_b
        self.r = best_r

        # LSH Hash tables: band_idx -> {bucket_hash -> list of sample_ids}
        self.tables: List[Dict[str, List[str]]] = [{} for _ in range(self.b)]
        self.signatures: Dict[str, List[int]] = {}

    def insert(self, key: str, text: str) -> bool:
        """Inserts text. Returns False if a near-duplicate above threshold already exists."""
        shingles = get_shingles(text)
        sig = self.minhash.compute_signature(shingles)

        # Check candidate near-duplicates across bands
        candidates: Set[str] = set()
        for i in range(self.b):
            band = sig[i * self.r : (i + 1) * self.r]
            band_hash = hashlib.sha256(str(band).encode("utf-8")).hexdigest()
            if band_hash in self.tables[i]:
                candidates.update(self.tables[i][band_hash])

        # Verify true Jaccard similarity with candidates
        for cand in candidates:
            cand_sig = self.signatures[cand]
            sim = MinHash.jaccard_similarity(sig, cand_sig)
            if sim >= self.threshold:
                return False  # Duplicate detected, reject insertion

        # No duplicate found, index signature
        self.signatures[key] = sig
        for i in range(self.b):
            band = sig[i * self.r : (i + 1) * self.r]
            band_hash = hashlib.sha256(str(band).encode("utf-8")).hexdigest()
            if band_hash not in self.tables[i]:
                self.tables[i][band_hash] = []
            self.tables[i][band_hash].append(key)

        return True
