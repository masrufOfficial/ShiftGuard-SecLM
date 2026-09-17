"""ShiftGuard-SecLM: Core Security Data Schemas.

Defines Pydantic data contracts for ShiftGuard-SecCorpus and ShiftGuard-SecBench
with strict field validation, ground-truth confidence levels, and provenance tracking.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    THREAT_ANALYSIS = "threat_analysis"
    SECURITY_REQUIREMENT = "security_requirement"
    CWE_CLASSIFY = "cwe_classify"
    OWASP_MAP = "owasp_map"
    RISK_ASSESS = "risk_assess"
    VULN_ANALYSIS = "vuln_analysis"
    SECURITY_PLAN = "security_plan"
    AGENT_PLAN = "agent_plan"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    REPAIR = "repair"
    VERIFY = "verify"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VerificationStatus(str, Enum):
    FIXED = "fixed"
    PARTIALLY_FIXED = "partially_fixed"
    STILL_VULNERABLE = "still_vulnerable"
    REGRESSION_INTRODUCED = "regression_introduced"


class SecurityFinding(BaseModel):
    source: str = Field(..., description="Tool name (e.g., semgrep, bandit, codeql, gitleaks)")
    rule_id: str = Field(..., description="Specific scanner rule identifier")
    cwe: Optional[str] = Field(None, description="Reported CWE identifier, e.g. CWE-89")
    severity: SeverityLevel = Field(SeverityLevel.MEDIUM, description="Finding severity")
    line: Optional[int] = Field(None, description="1-indexed line number in source code")
    message: str = Field(..., description="Explanation from scanner")
    raw_payload: Optional[Dict[str, Any]] = None


class RiskAssessment(BaseModel):
    severity: SeverityLevel = Field(..., description="Calculated overall severity")
    score: float = Field(..., ge=0.0, le=10.0, description="CVSS-aligned risk score 0.0-10.0")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model calibration confidence 0.0-1.0")
    explanation: str = Field(..., description="Contextual rationale for risk rating")


class AgentStep(BaseModel):
    agent: str = Field(..., description="Recommended security agent type (e.g. auth_agent, injection_agent)")
    action: str = Field(..., description="Specific investigation or verification action")
    priority: int = Field(1, ge=1, le=5, description="Execution priority from 1 (highest) to 5")


class SecurityGroundTruth(BaseModel):
    threats: List[str] = Field(default_factory=list, description="Identified threat names")
    cwe: List[str] = Field(default_factory=list, description="Associated CWE codes, e.g. ['CWE-89']")
    owasp: List[str] = Field(default_factory=list, description="Associated OWASP categories, e.g. ['A03:2021-Injection']")
    risk: Optional[RiskAssessment] = None
    security_requirements: List[str] = Field(default_factory=list, description="Extracted defensive requirements")
    security_strategy: List[str] = Field(default_factory=list, description="Strategic architectural recommendations")
    recommended_capabilities: List[str] = Field(default_factory=list, description="Required ShiftGuard capability modules")
    agent_plan: List[AgentStep] = Field(default_factory=list, description="Multi-agent orchestration sequence")
    explanation: Optional[str] = Field(None, description="Detailed security reasoning chain")
    repair_guidance: List[str] = Field(default_factory=list, description="Step-by-step remediation advice")
    repair_code: Optional[str] = Field(None, description="Secure patch or complete remediated source code")
    verification: Optional[VerificationStatus] = None


class ProvenanceMetadata(BaseModel):
    source_name: str = Field(..., description="Source dataset (e.g. cvefixes, juliet, owasp_benchmark, synthetic)")
    source_id: Optional[str] = None
    repository: Optional[str] = None
    commit_hash: Optional[str] = None
    license: str = Field("Apache-2.0", description="Data license")
    is_synthetic: bool = False
    archetype: Optional[str] = None
    split: Optional[str] = Field(None, description="train, validation, or test")


class SecuritySample(BaseModel):
    """Canonical instance representing a single security reasoning sample in ShiftGuard datasets."""
    sample_id: str = Field(..., description="Unique sample ID (e.g. SG-CORPUS-001024)")
    task: TaskType = Field(..., description="Task category from Tasks 1-11")
    requirement: Optional[str] = Field(None, description="Software requirement specification")
    prompt: Optional[str] = Field(None, description="Developer prompt input")
    context: Dict[str, Any] = Field(default_factory=dict, description="Project context (lang, framework, db, etc.)")
    code: Optional[str] = Field(None, description="Source code snippet under analysis")
    security_findings: List[SecurityFinding] = Field(default_factory=list, description="Findings from scanners")
    ground_truth: SecurityGroundTruth = Field(..., description="Structured ground truth")
    confidence: ConfidenceLevel = Field(ConfidenceLevel.HIGH, description="Ground truth confidence")
    metadata: ProvenanceMetadata = Field(..., description="Provenance and licensing metadata")

    def to_scf_payload(self) -> Dict[str, Any]:
        """Converts to dictionary compatible with SecurityContextPayload."""
        target_dict = self.ground_truth.model_dump(exclude_none=True)
        return {
            "task": self.task.value,
            "requirement": self.requirement,
            "prompt": self.prompt,
            "language": self.context.get("language"),
            "framework": self.context.get("framework"),
            "code": self.code,
            "security_findings": [f.model_dump() for f in self.security_findings],
            "project_metadata": self.context,
            "target_output": target_dict,
        }

