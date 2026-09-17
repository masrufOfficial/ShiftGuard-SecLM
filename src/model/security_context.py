"""Security Context Fusion (SCF) Module for ShiftGuard-SecLM.

Fuses heterogeneous software security modalities (natural-language requirements,
developer prompts, programming language, framework metadata, source code, and
scanner findings) into a canonical structured representation with explicit task conditioning.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn


# Task tokens defined in Master Specification Section 9 & 11
TASK_TOKENS = {
    "threat_analysis": "<TASK:THREAT_ANALYSIS>",
    "security_requirement": "<TASK:SECURITY_REQUIREMENT>",
    "cwe_classify": "<TASK:CWE_CLASSIFY>",
    "owasp_map": "<TASK:OWASP_MAP>",
    "risk_assess": "<TASK:RISK_ASSESS>",
    "vuln_analysis": "<TASK:VULN_ANALYSIS>",
    "security_plan": "<TASK:SECURITY_PLAN>",
    "agent_plan": "<TASK:AGENT_PLAN>",
    "evidence_analysis": "<TASK:EVIDENCE_ANALYSIS>",
    "repair": "<TASK:REPAIR>",
    "verify": "<TASK:VERIFY>",
}

BOUNDARY_TOKENS = {
    "context_start": "<SEC_CONTEXT>",
    "context_end": "</SEC_CONTEXT>",
    "schema_start": "<SCHEMA_START>",
    "schema_end": "<SCHEMA_END>",
}


@dataclass
class SecurityContextPayload:
    """Structured input container for Security Context Fusion."""
    task: str
    requirement: Optional[str] = None
    prompt: Optional[str] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    code: Optional[str] = None
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    project_metadata: Dict[str, Any] = field(default_factory=dict)
    target_output: Optional[Dict[str, Any]] = None

    def serialize_input(self) -> str:
        """Serializes input metadata, code, and evidence into canonical text."""
        parts = [BOUNDARY_TOKENS["context_start"]]

        if self.language:
            parts.append(f"<LANG:{self.language.lower().strip()}>")
        if self.framework:
            parts.append(f"<FRAMEWORK:{self.framework.lower().strip()}>")
        if self.requirement:
            parts.append(f"<REQUIREMENT:{self.requirement.strip()}>")
        if self.prompt:
            parts.append(f"<PROMPT:{self.prompt.strip()}>")
        if self.project_metadata:
            meta_str = json.dumps(self.project_metadata, separators=(",", ":"))
            parts.append(f"<PROJECT_META:{meta_str}>")
        if self.security_findings:
            findings_str = json.dumps(self.security_findings, separators=(",", ":"))
            parts.append(f"<EVIDENCE:{findings_str}>")
        if self.code:
            parts.append(f"<CODE>\n{self.code}\n</CODE>")

        parts.append(BOUNDARY_TOKENS["context_end"])

        # Append task token
        task_token = TASK_TOKENS.get(self.task.lower(), f"<TASK:{self.task.upper()}>")
        parts.append(task_token)
        parts.append(BOUNDARY_TOKENS["schema_start"])

        return " ".join(parts)

    def serialize_full_sequence(self) -> Tuple[str, str]:
        """Returns (prompt_text, completion_text) for loss-masked training."""
        input_text = self.serialize_input()
        if self.target_output is not None:
            output_json = json.dumps(self.target_output, indent=2)
            completion = f"\n{output_json}\n{BOUNDARY_TOKENS['schema_end']}"
        else:
            completion = ""
        return input_text, completion


class SecurityContextFusion(nn.Module):
    """Neural Security Context Fusion layer.
    
    Provides optional segment-type embeddings to demarcate context metadata
    from executable source code and reasoning output tokens.
    """

    def __init__(self, d_model: int, num_segments: int = 4):
        super().__init__()
        # Segment 0: Special/Control tokens
        # Segment 1: Metadata / Requirements
        # Segment 2: Source Code
        # Segment 3: Structured Reasoning / Output
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if segment_ids is None:
            return token_embeddings
        seg_emb = self.segment_embedding(segment_ids)
        return self.layer_norm(token_embeddings + seg_emb)

