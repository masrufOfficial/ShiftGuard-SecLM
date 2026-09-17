"""Custom BPE Tokenizer Pipeline for ShiftGuard-SecLM.

Constructs, trains, and exports a domain-specialized Byte-Pair Encoding (BPE)
tokenizer with 0% OOV via Byte-Level pre-tokenization and reserved security tokens.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from src.model.security_context import TASK_TOKENS, BOUNDARY_TOKENS

# Base control tokens
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
]

# Add all task and boundary tokens
ALL_SPECIAL_TOKENS = (
    SPECIAL_TOKENS
    + list(BOUNDARY_TOKENS.values())
    + list(TASK_TOKENS.values())
    + [
        "<CODE>",
        "</CODE>",
        "<LANG:python>",
        "<LANG:javascript>",
        "<LANG:typescript>",
        "<LANG:java>",
        "<LANG:c>",
        "<LANG:cpp>",
        "<LANG:go>",
        "<LANG:php>",
        "<LANG:rust>",
        "<LANG:ruby>",
        "<LANG:csharp>",
        "<LANG:sql>",
        "<FRAMEWORK:fastapi>",
        "<FRAMEWORK:flask>",
        "<FRAMEWORK:django>",
        "<FRAMEWORK:express>",
        "<FRAMEWORK:spring>",
        "<FRAMEWORK:actix>",
        "<FRAMEWORK:rails>",
        "<FRAMEWORK:aspnet>",
        "<FRAMEWORK:gin>",
        "<FRAMEWORK:standard_library>",
    ]
)


def create_base_tokenizer() -> Tokenizer:
    """Instantiates a Byte-Level BPE tokenizer with robust pre-tokenization."""
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    # ByteLevel pre-tokenizer splits on whitespace & punctuation while preserving exact byte values
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def train_security_tokenizer(
    training_files: List[str],
    vocab_size: int = 32000,
    min_frequency: int = 2,
    output_dir: Optional[str | Path] = None,
) -> Tokenizer:
    """Trains a domain-specialized BPE tokenizer from raw text files."""
    tokenizer = create_base_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=ALL_SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files=training_files, trainer=trainer)

    if output_dir is not None:
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path / "tokenizer.json"))
        # Save vocabulary map
        vocab = tokenizer.get_vocab()
        with open(save_path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2)

    return tokenizer


def test_tokenizer_efficiency(tokenizer: Tokenizer, test_samples: Dict[str, str]) -> Dict[str, Any]:
    """Evaluates tokenizer compression efficiency across code and security terminology."""
    results = {}
    for category, text in test_samples.items():
        encoded = tokenizer.encode(text)
        raw_bytes = len(text.encode("utf-8"))
        num_tokens = len(encoded.ids)
        compression_ratio = raw_bytes / max(num_tokens, 1)
        tokens_preview = encoded.tokens[:15]
        results[category] = {
            "num_chars": len(text),
            "num_bytes": raw_bytes,
            "num_tokens": num_tokens,
            "bytes_per_token": round(compression_ratio, 2),
            "tokens_preview": tokens_preview,
        }
    return results

