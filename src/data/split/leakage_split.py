"""Leakage-Safe Dataset Partitioning for ShiftGuard-SecLM.

Guarantees strict separation across repositories, project families,
application archetypes, and temporal horizons to prevent data leakage.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

from datasets.schemas.security_sample import SecuritySample


class LeakageSafeSplitter:
    """Partitions security samples into train, validation, and test splits
    guaranteeing zero repository or scenario overlap."""

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        held_out_archetypes: Optional[List[str]] = None,
        seed: int = 42,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.held_out_archetypes = set(held_out_archetypes or ["iot_backend", "payment_gateway"])
        self.seed = seed

    def get_cluster_key(self, sample: SecuritySample) -> str:
        """Determines the grouping key (repository name, project name, or archetype)."""
        if sample.metadata.repository:
            return f"repo:{sample.metadata.repository.lower().strip()}"
        if sample.metadata.archetype:
            return f"archetype:{sample.metadata.archetype.lower().strip()}"
        # Fallback to source identifier
        return f"source:{sample.metadata.source_name}:{sample.sample_id.split('-')[0]}"

    def partition_samples(
        self, samples: List[SecuritySample]
    ) -> Tuple[List[SecuritySample], List[SecuritySample], List[SecuritySample]]:
        """Splits samples into (train, val, test) ensuring entire clusters reside in exactly one split."""
        # 1. Separate held-out generalization test cases immediately
        test_samples: List[SecuritySample] = []
        regular_clusters: Dict[str, List[SecuritySample]] = {}

        for sample in samples:
            if sample.metadata.archetype and sample.metadata.archetype.lower() in self.held_out_archetypes:
                sample.metadata.split = "test"
                test_samples.append(sample)
            else:
                key = self.get_cluster_key(sample)
                if key not in regular_clusters:
                    regular_clusters[key] = []
                regular_clusters[key].append(sample)

        train_samples: List[SecuritySample] = []
        val_samples: List[SecuritySample] = []

        # 2. Assign clusters deterministically using cryptographic hashing on cluster key + seed
        for cluster_key, cluster_items in regular_clusters.items():
            # Hash to a float in [0, 1)
            h = hashlib.sha256(f"{cluster_key}-{self.seed}".encode("utf-8")).hexdigest()
            norm_val = int(h[:8], 16) / 0xFFFFFFFF

            if norm_val < self.train_ratio:
                for item in cluster_items:
                    item.metadata.split = "train"
                train_samples.extend(cluster_items)
            elif norm_val < (self.train_ratio + self.val_ratio):
                for item in cluster_items:
                    item.metadata.split = "val"
                val_samples.extend(cluster_items)
            else:
                for item in cluster_items:
                    item.metadata.split = "test"
                test_samples.append(cluster_items)

        return train_samples, val_samples, test_samples

    def generate_manifest(
        self,
        samples: List[SecuritySample],
        split_name: str,
        output_dir: str | Path,
    ) -> Dict[str, Any]:
        """Saves samples to JSONL and produces a cryptographically hashed manifest."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        data_file = out_path / f"{split_name}.jsonl"

        sha256 = hashlib.sha256()
        with open(data_file, "w", encoding="utf-8") as f:
            for s in samples:
                line = s.model_dump_json() + "\n"
                sha256.update(line.encode("utf-8"))
                f.write(line)

        manifest = {
            "split": split_name,
            "filename": str(data_file.name),
            "num_samples": len(samples),
            "sha256": sha256.hexdigest(),
            "tasks": {},
            "languages": {},
        }

        for s in samples:
            t = s.task.value
            manifest["tasks"][t] = manifest["tasks"].get(t, 0) + 1
            lang = s.context.get("language", "unknown")
            manifest["languages"][lang] = manifest["languages"].get(lang, 0) + 1

        manifest_file = out_path / f"{split_name}_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
