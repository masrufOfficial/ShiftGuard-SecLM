"""Evaluation Metrics for ShiftGuard-SecLM.

Implements rigorous security evaluation metrics:
- Multi-label CWE Precision, Recall, and Macro/Micro-F1
- OWASP Mapping Accuracy
- Risk Calibration: Brier Score and Expected Calibration Error (ECE)
- JSON Schema Compliance & Syntax Validity
- Security Repair Pass Rate
"""

from __future__ import annotations
import json
import numpy as np
from typing import List, Dict, Set, Any, Tuple


def calculate_multilabel_f1(
    predictions: List[List[str]],
    references: List[List[str]],
) -> Dict[str, float]:
    """Calculates macro and micro Precision, Recall, and F1 across multi-label CWE predictions."""
    assert len(predictions) == len(references)
    
    all_classes: Set[str] = set()
    for preds, refs in zip(predictions, references):
        all_classes.update(preds)
        all_classes.update(refs)

    if not all_classes:
        return {"precision_micro": 1.0, "recall_micro": 1.0, "f1_micro": 1.0, "f1_macro": 1.0}

    # Micro counts
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Macro collectors
    per_class_f1: List[float] = []

    for cls in all_classes:
        tp, fp, fn = 0, 0, 0
        for preds, refs in zip(predictions, references):
            p_has = cls in preds
            r_has = cls in refs
            if p_has and r_has:
                tp += 1
            elif p_has and not r_has:
                fp += 1
            elif not p_has and r_has:
                fn += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)

    p_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = (2 * p_micro * r_micro) / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0
    f1_macro = float(np.mean(per_class_f1))

    return {
        "precision_micro": round(p_micro, 4),
        "recall_micro": round(r_micro, 4),
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "num_classes": len(all_classes),
    }


def calculate_brier_score(
    predicted_probabilities: List[float],
    binary_outcomes: List[int],
) -> float:
    """Calculates Brier Score for probability calibration: (1/N) * sum((p_i - y_i)^2)."""
    assert len(predicted_probabilities) == len(binary_outcomes)
    if not predicted_probabilities:
        return 0.0
    p = np.array(predicted_probabilities)
    y = np.array(binary_outcomes)
    return float(np.mean((p - y) ** 2))


def calculate_expected_calibration_error(
    predicted_probabilities: List[float],
    binary_outcomes: List[int],
    num_bins: int = 10,
) -> float:
    """Calculates Expected Calibration Error (ECE) across confidence bins."""
    assert len(predicted_probabilities) == len(binary_outcomes)
    if not predicted_probabilities:
        return 0.0

    p = np.array(predicted_probabilities)
    y = np.array(binary_outcomes)
    n = len(p)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (p > bin_lower) & (p <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y[in_bin])
            avg_confidence_in_bin = np.mean(p[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return round(float(ece), 4)


def validate_json_schema(generated_text: str, expected_keys: Optional[List[str]] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Tests whether generated completion parses as valid JSON with required security fields."""
    expected_keys = expected_keys or ["threats", "cwe", "owasp", "risk"]
    try:
        # Strip potential markdown formatting
        text = generated_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        if not isinstance(data, dict):
            return False, None

        for k in expected_keys:
            if k not in data:
                return False, data
        return True, data
    except Exception:
        return False, None
