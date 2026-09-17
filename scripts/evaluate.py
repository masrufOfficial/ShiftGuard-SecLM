#!/usr/bin/env python3
"""ShiftGuard-SecLM: Benchmark Evaluation CLI Script.

Evaluates trained model checkpoints across multi-label CWE classification,
OWASP mapping, risk calibration, and structured JSON schema compliance.
"""

import sys
import json
import argparse
from pathlib import Path

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tokenizers import Tokenizer

from src.model.security_context import SecurityContextPayload
from src.training.checkpointing import load_checkpoint
from src.evaluation.metrics import (
    calculate_multilabel_f1,
    calculate_brier_score,
    calculate_expected_calibration_error,
    validate_json_schema,
)


def evaluate():
    parser = argparse.ArgumentParser(description="ShiftGuard-SecLM Benchmark Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint step directory")
    parser.add_argument("--test-data", type=str, default="datasets/manifests/test.jsonl", help="Test dataset JSONL")
    parser.add_argument("--tokenizer-path", type=str, default="datasets/processed/tokenizer/tokenizer.json", help="Tokenizer JSON path")
    parser.add_argument("--output-file", type=str, default="experiments/eval_results.json", help="Output results file")
    parser.add_argument("--max-samples", type=int, default=20, help="Max test samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    print("=" * 80)
    print(" SHIFTGUARD-SECLM: BENCHMARK EVALUATION SUITE")
    print("=" * 80)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Test Data:      {args.test_data}")
    print(f"Device:         {args.device}")

    # 1. Load Tokenizer & Model
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    model, config, _ = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    print(f"Model {config.model_name} loaded from {args.checkpoint}.")

    # 2. Read Test Samples
    test_path = Path(args.test_data)
    test_samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
            if len(test_samples) >= args.max_samples:
                break

    print(f"Evaluating on {len(test_samples)} held-out test scenarios...")

    cwe_preds: list[list[str]] = []
    cwe_refs: list[list[str]] = []
    owasp_matches = 0
    valid_json_count = 0
    risk_probs: list[float] = []
    risk_targets: list[int] = []

    for i, s in enumerate(test_samples):
        gt = s.get("ground_truth", {})
        payload = SecurityContextPayload(
            task=s.get("task", "threat_analysis"),
            requirement=s.get("requirement"),
            prompt=s.get("prompt"),
            language=s.get("context", {}).get("language"),
            framework=s.get("context", {}).get("framework"),
            code=s.get("code"),
            security_findings=s.get("security_findings", []),
            project_metadata=s.get("context", {}),
        )

        prompt_str = payload.serialize_input()
        input_ids = torch.tensor([tokenizer.encode(prompt_str).ids], device=args.device)

        with torch.no_grad():
            gen_ids = model.generate(input_ids, max_new_tokens=64, temperature=0.0)

        gen_tokens = gen_ids[0][input_ids.shape[1] :].tolist()
        gen_text = tokenizer.decode(gen_tokens)

        # 1. Schema Validation
        is_valid, parsed_data = validate_json_schema(gen_text)
        if is_valid:
            valid_json_count += 1

        # 2. Ground truth comparisons
        ref_cwes = gt.get("cwe", [])
        cwe_refs.append(ref_cwes)

        pred_cwes = parsed_data.get("cwe", []) if is_valid else []
        cwe_preds.append(pred_cwes)

        ref_owasp = set(gt.get("owasp", []))
        pred_owasp = set(parsed_data.get("owasp", [])) if is_valid else set()
        if ref_owasp and ref_owasp.intersection(pred_owasp):
            owasp_matches += 1

        # Risk calibration
        conf = parsed_data.get("risk", {}).get("confidence", 0.5) if is_valid else 0.5
        risk_probs.append(float(conf))
        is_high_risk = 1 if gt.get("risk", {}).get("severity") in ["critical", "high"] else 0
        risk_targets.append(is_high_risk)

    # Calculate overall metrics
    f1_metrics = calculate_multilabel_f1(cwe_preds, cwe_refs)
    brier = calculate_brier_score(risk_probs, risk_targets)
    ece = calculate_expected_calibration_error(risk_probs, risk_targets)
    json_validity_rate = valid_json_count / max(len(test_samples), 1)
    owasp_accuracy = owasp_matches / max(len(test_samples), 1)

    results = {
        "num_test_samples": len(test_samples),
        "json_schema_validity_rate": round(json_validity_rate, 4),
        "owasp_mapping_accuracy": round(owasp_accuracy, 4),
        "cwe_f1_macro": f1_metrics["f1_macro"],
        "cwe_f1_micro": f1_metrics["f1_micro"],
        "cwe_precision_micro": f1_metrics["precision_micro"],
        "cwe_recall_micro": f1_metrics["recall_micro"],
        "risk_brier_score": round(brier, 4),
        "risk_expected_calibration_error": ece,
    }

    print("\n--- BENCHMARK RESULTS SUMMARY ---")
    print(f"  * JSON Schema Validity Rate:        {results['json_schema_validity_rate']*100:.1f}%")
    print(f"  * OWASP Mapping Accuracy:           {results['owasp_mapping_accuracy']*100:.1f}%")
    print(f"  * CWE Multi-label F1 (Macro):       {results['cwe_f1_macro']:.4f}")
    print(f"  * CWE Multi-label F1 (Micro):       {results['cwe_f1_micro']:.4f}")
    print(f"  * Risk Calibration (Brier Score):   {results['risk_brier_score']:.4f}")
    print(f"  * Expected Calibration Error (ECE): {results['risk_expected_calibration_error']:.4f}")

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results saved to {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate()

