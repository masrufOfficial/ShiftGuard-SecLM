"""ShiftGuard-SecLM: FastAPI Inference Service.

Exposes the standardized /analyze API contract defined in Master Specification Section 17 & 23,
allowing the future ShiftGuard multi-agent framework to query this security reasoning model.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from src.model.security_context import SecurityContextPayload
from src.training.checkpointing import load_checkpoint
from src.evaluation.metrics import validate_json_schema


# Request & Response Data Transfer Objects (DTOs)
class AnalyzeRequest(BaseModel):
    task: str = Field("threat_analysis", description="Security reasoning task category")
    requirement: Optional[str] = Field(None, description="Software functional requirement")
    prompt: Optional[str] = Field(None, description="Developer prompt or question")
    language: Optional[str] = Field("python", description="Programming language")
    framework: Optional[str] = Field("fastapi", description="Web/Application framework")
    code: Optional[str] = Field(None, description="Source code snippet")
    security_findings: List[Dict[str, Any]] = Field(default_factory=list, description="Scanner findings")
    project_context: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class RiskDTO(BaseModel):
    severity: str = "medium"
    score: float = 5.0
    confidence: float = 0.5
    explanation: Optional[str] = None


class AnalyzeResponse(BaseModel):
    threats: List[str] = Field(default_factory=list)
    cwe: List[str] = Field(default_factory=list)
    owasp: List[str] = Field(default_factory=list)
    risk: RiskDTO = Field(default_factory=RiskDTO)
    security_requirements: List[str] = Field(default_factory=list)
    security_strategy: List[str] = Field(default_factory=list)
    recommended_capabilities: List[str] = Field(default_factory=list)
    agent_plan: List[Dict[str, Any]] = Field(default_factory=list)
    explanation: str = ""
    repair_guidance: List[str] = Field(default_factory=list)
    repair_code: Optional[str] = None
    verification: Optional[str] = None
    confidence: Dict[str, float] = Field(default_factory=dict)


# FastAPI Application
app = FastAPI(
    title="ShiftGuard-SecLM Inference Service",
    description="Contextual Shift-Left Security Reasoning API for AI-Assisted Software Development",
    version="0.1.0",
)

MODEL_HOLDER = {}


@app.on_event("startup")
def load_model_on_startup():
    default_ckpt = "experiments/run_01/step_0000058"
    run_dir = Path("experiments/run_01")
    if run_dir.exists():
        ckpts = sorted(run_dir.glob("step_*"))
        if ckpts:
            default_ckpt = str(ckpts[-1])

    ckpt_dir = os.environ.get("SHIFTGUARD_CHECKPOINT", default_ckpt)
    tok_dir = os.environ.get("SHIFTGUARD_TOKENIZER", "datasets/processed/tokenizer/tokenizer.json")

    if Path(ckpt_dir).exists() and Path(tok_dir).exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = Tokenizer.from_file(tok_dir)
        model, config, _ = load_checkpoint(ckpt_dir, device=device)
        model.eval()
        MODEL_HOLDER["model"] = model
        MODEL_HOLDER["tokenizer"] = tokenizer
        MODEL_HOLDER["device"] = device
        print(f"Loaded ShiftGuard-SecLM from {ckpt_dir} on {device}")


@app.get("/health")
def health_check():
    loaded = "model" in MODEL_HOLDER
    return {"status": "ok" if loaded else "unloaded", "model": loaded}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_security_context(request: AnalyzeRequest):
    if "model" not in MODEL_HOLDER:
        raise HTTPException(status_code=503, detail="ShiftGuard-SecLM model not loaded.")

    model = MODEL_HOLDER["model"]
    tokenizer = MODEL_HOLDER["tokenizer"]
    device = MODEL_HOLDER["device"]

    payload = SecurityContextPayload(
        task=request.task,
        requirement=request.requirement,
        prompt=request.prompt,
        language=request.language,
        framework=request.framework,
        code=request.code,
        security_findings=request.security_findings,
        project_metadata=request.project_context,
    )

    prompt_str = payload.serialize_input()
    encoded_ids = tokenizer.encode(prompt_str).ids
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        v_limit = model.config.vocab_size
        unk_id = tokenizer.token_to_id("<|unk|>") or 0
        encoded_ids = [tid if tid < v_limit else unk_id for tid in encoded_ids]
    input_ids = torch.tensor([encoded_ids], device=device)

    with torch.no_grad():
        gen_ids = model.generate(input_ids, max_new_tokens=128, temperature=0.2)

    gen_tokens = gen_ids[0][input_ids.shape[1] :].tolist()
    gen_text = tokenizer.decode(gen_tokens)

    is_valid, data = validate_json_schema(gen_text)
    if is_valid and data:
        return AnalyzeResponse(**data)
    else:
        # Fallback structured response
        return AnalyzeResponse(
            threats=[],
            cwe=[],
            owasp=[],
            risk=RiskDTO(severity="medium", score=5.0, confidence=0.5, explanation=gen_text),
            explanation=gen_text,
        )

