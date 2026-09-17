"""ShiftGuard-SecLM: Research-Grade Security Data Schemas (Schema V2.0.0).

Defines Pydantic data contracts for ShiftGuard-SecCorpus and ShiftGuard-SecBench
with strict field validation, ground-truth confidence levels, multi-modal context,
provenance tracking, and quality flags supporting all 12 ShiftGuard security reasoning tasks.
"""

from __future__ import annotations
import re
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    THREAT_ANALYSIS = "threat_analysis"
    SECURITY_REQUIREMENT = "security_requirement"
    CWE_CLASSIFY = "cwe_classify"
    OWASP_MAP = "owasp_map"
    RISK_ASSESS = "risk_assess"
    VULN_ANALYSIS = "vuln_analysis"
    SECURITY_PLAN = "security_plan"
    CAPABILITY_PLAN = "capability_plan"
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


class ProvenanceMetadata(BaseModel):
    source_name: str = Field(..., description="Source dataset (e.g. cvefixes, juliet, owasp_benchmark, synthetic)")
    source_id: Optional[str] = None
    repository: Optional[str] = None
    commit_hash: Optional[str] = None
    license: str = Field("Apache-2.0", description="Data license")
    is_synthetic: bool = False
    evidence_source: Optional[str] = Field(None, description="Known CWE, NVD CVE, Juliet, SAST finding, or Manual")
    author: Optional[str] = None
    archetype: Optional[str] = None
    split: Optional[str] = Field(None, description="train, validation, or test")


class QualityAuditFlags(BaseModel):
    """Container for automated quality audit checks on a sample."""
    valid_json: bool = True
    valid_schema: bool = True
    non_empty_code: bool = True
    non_empty_prompt: bool = True
    valid_cwe: bool = True
    valid_owasp: bool = True
    has_provenance: bool = True
    has_license: bool = True
    consistent_severity: bool = True


class SecurityGroundTruth(BaseModel):
    threats: List[str] = Field(default_factory=list, description="Identified threat names")
    cwe: List[str] = Field(default_factory=list, description="Associated CWE codes, e.g. ['CWE-89']")
    owasp: List[str] = Field(default_factory=list, description="Associated OWASP categories, e.g. ['A03:2021-Injection']")
    risk: Optional[RiskAssessment] = None
    security_requirements: List[str] = Field(default_factory=list, description="Extracted defensive requirements")
    security_strategy: List[str] = Field(default_factory=list, description="Strategic architectural recommendations")
    capability_plan: List[str] = Field(default_factory=list, description="Required ShiftGuard capability modules")
    recommended_capabilities: List[str] = Field(default_factory=list, description="Alias for capability_plan")
    agent_plan: List[AgentStep] = Field(default_factory=list, description="Multi-agent orchestration sequence")
    explanation: Optional[str] = Field(None, description="Detailed security reasoning chain")
    repair_guidance: List[str] = Field(default_factory=list, description="Step-by-step remediation advice")
    repair_code: Optional[str] = Field(None, description="Secure patch or complete remediated source code")
    verification: Optional[VerificationStatus] = None


class SecuritySample(BaseModel):
    """Canonical research instance in ShiftGuard-SecCorpus (Schema V2.0.0).
    
    Supports all 12 ShiftGuard security reasoning tasks with full provenance,
    ground-truth evidence, and quality audit flags.
    """
    schema_version: str = Field("2.0.0", description="Data schema specification version")
    sample_id: str = Field(..., description="Unique sample identifier (e.g. SG-CORPUS-001024)")
    source: str = Field(..., description="Origin dataset or generator (e.g., juliet, cvefixes, synthetic_shiftguard)")
    source_id: Optional[str] = Field(None, description="Underlying upstream identifier or CVE-ID")
    application_archetype: Optional[str] = Field(None, description="Target application domain / archetype")
    domain: Optional[str] = Field(None, description="High-level industry or tech domain")
    repository: Optional[str] = Field(None, description="Repository or project identifier for leakage grouping")
    language: Optional[str] = Field(None, description="Programming language (e.g., python, javascript, c)")
    framework: Optional[str] = Field(None, description="Framework or runtime (e.g., fastapi, express, spring)")
    database: Optional[str] = Field(None, description="Database type (e.g., postgresql, sqlite, mongodb)")
    api_type: Optional[str] = Field(None, description="API protocol (e.g., REST, GraphQL, gRPC, CLI)")

    # Context & Code Modalities
    requirement: Optional[str] = Field(None, description="Software requirement specification")
    developer_prompt: Optional[str] = Field(None, description="Developer query or code completion prompt")
    project_context: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")
    source_code: Optional[str] = Field(None, description="Vulnerable or target source code")
    security_context: Optional[str] = Field(None, description="Raw serialized security context tokens")
    security_findings: List[SecurityFinding] = Field(default_factory=list, description="Findings from scanners")

    # Security Ground Truth & Reasoning Targets
    threat_description: Optional[str] = Field(None, description="Detailed threat narrative")
    security_requirements: List[str] = Field(default_factory=list, description="Defensive security requirements")
    cwe: List[str] = Field(default_factory=list, description="CWE IDs (e.g. ['CWE-89'])")
    cwe_description: Optional[str] = Field(None, description="Official MITRE CWE title/description")
    owasp_category: List[str] = Field(default_factory=list, description="OWASP categories")
    risk_level: Optional[str] = Field(None, description="Categorical risk rating: critical, high, medium, low")
    severity: Optional[str] = Field(None, description="Vulnerability severity level")
    confidence: ConfidenceLevel = Field(ConfidenceLevel.HIGH, description="Ground truth confidence")
    security_strategy: List[str] = Field(default_factory=list, description="Architectural defense strategy")
    capability_plan: List[str] = Field(default_factory=list, description="Capability modules required")
    agent_plan: List[AgentStep] = Field(default_factory=list, description="Multi-agent orchestration sequence")
    evidence: Optional[str] = Field(None, description="Verification evidence or scanner output")
    vulnerability_explanation: Optional[str] = Field(None, description="Contextual reasoning explanation")
    secure_repair: Optional[str] = Field(None, description="Remediated code or security patch")
    verification: Optional[VerificationStatus] = Field(None, description="Verification status of repair")

    # Task & Partitioning
    task_type: TaskType = Field(..., description="Target reasoning task from Tasks 1-12")
    split: Optional[str] = Field(None, description="train, validation, or test split")
    provenance: ProvenanceMetadata = Field(..., description="Provenance, attribution and licensing")
    license: str = Field("Apache-2.0", description="Redistribution license")
    quality_flags: Dict[str, bool] = Field(default_factory=dict, description="Data validation quality checks")

    # Backward compatibility aliases for existing loaders
    prompt: Optional[str] = None
    code: Optional[str] = None
    task: Optional[TaskType] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[ProvenanceMetadata] = None
    ground_truth: Optional[SecurityGroundTruth] = None

    @model_validator(mode="before")
    @classmethod
    def sync_v1_and_v2_fields(cls, data: Any) -> Any:
        """Bi-directionally synchronizes V1 and V2 fields for seamless compatibility."""
        if not isinstance(data, dict):
            return data

        # 1. Sync source_code <-> code
        if "source_code" in data and data["source_code"] and not data.get("code"):
            data["code"] = data["source_code"]
        elif "code" in data and data["code"] and not data.get("source_code"):
            data["source_code"] = data["code"]

        # 2. Sync developer_prompt <-> prompt
        if "developer_prompt" in data and data["developer_prompt"] and not data.get("prompt"):
            data["prompt"] = data["developer_prompt"]
        elif "prompt" in data and data["prompt"] and not data.get("developer_prompt"):
            data["developer_prompt"] = data["prompt"]

        # 3. Sync task_type <-> task
        if "task_type" in data and data["task_type"] and not data.get("task"):
            data["task"] = data["task_type"]
        elif "task" in data and data["task"] and not data.get("task_type"):
            data["task_type"] = data["task"]

        # 4. Sync project_context <-> context
        if "project_context" in data and data["project_context"] and not data.get("context"):
            data["context"] = data["project_context"]
        elif "context" in data and data["context"] and not data.get("project_context"):
            data["project_context"] = data["context"]

        # 5. Language & Framework extraction into root if in context
        ctx = data.get("context") or data.get("project_context") or {}
        if not data.get("language") and "language" in ctx:
            data["language"] = ctx["language"]
        if not data.get("framework") and "framework" in ctx:
            data["framework"] = ctx["framework"]
        if not data.get("database") and "database" in ctx:
            data["database"] = ctx["database"]
        if not data.get("api_type") and "api_type" in ctx:
            data["api_type"] = ctx["api_type"]

        # 6. Sync metadata <-> provenance
        if "metadata" in data and data["metadata"] and not data.get("provenance"):
            data["provenance"] = data["metadata"]
        elif "provenance" in data and data["provenance"] and not data.get("metadata"):
            data["metadata"] = data["provenance"]

        # 7. Extract source, repository, archetype from provenance if not set
        prov = data.get("provenance") or data.get("metadata") or {}
        prov_dict = prov.model_dump() if hasattr(prov, "model_dump") else (prov if isinstance(prov, dict) else {})
        if prov_dict:
            if not data.get("source") and "source_name" in prov_dict:
                data["source"] = prov_dict["source_name"]
            if not data.get("source_id") and "source_id" in prov_dict:
                data["source_id"] = prov_dict["source_id"]
            if not data.get("repository") and "repository" in prov_dict:
                data["repository"] = prov_dict["repository"]
            if not data.get("application_archetype") and "archetype" in prov_dict:
                data["application_archetype"] = prov_dict["archetype"]
            if not data.get("license") and "license" in prov_dict:
                data["license"] = prov_dict["license"]
            if not data.get("split") and "split" in prov_dict:
                data["split"] = prov_dict["split"]

        if not data.get("source"):
            data["source"] = "shiftguard_corpus"

        # 8. Ensure ground_truth is populated or sync fields into ground_truth
        gt = data.get("ground_truth") or {}
        gt_dict = gt.model_dump() if hasattr(gt, "model_dump") else (gt if isinstance(gt, dict) else {})
        if gt_dict:
            if not data.get("cwe") and "cwe" in gt_dict:
                data["cwe"] = gt_dict["cwe"]
            if not data.get("owasp_category") and "owasp" in gt_dict:
                data["owasp_category"] = gt_dict["owasp"]
            if not data.get("security_requirements") and "security_requirements" in gt_dict:
                data["security_requirements"] = gt_dict["security_requirements"]
            if not data.get("security_strategy") and "security_strategy" in gt_dict:
                data["security_strategy"] = gt_dict["security_strategy"]
            if not data.get("capability_plan") and "capability_plan" in gt_dict:
                data["capability_plan"] = gt_dict["capability_plan"]
            elif not data.get("capability_plan") and "recommended_capabilities" in gt_dict:
                data["capability_plan"] = gt_dict["recommended_capabilities"]
            if not data.get("agent_plan") and "agent_plan" in gt_dict:
                data["agent_plan"] = gt_dict["agent_plan"]
            if not data.get("secure_repair") and "repair_code" in gt_dict:
                data["secure_repair"] = gt_dict["repair_code"]
            if not data.get("verification") and "verification" in gt_dict:
                data["verification"] = gt_dict["verification"]
            if not data.get("vulnerability_explanation") and "explanation" in gt_dict:
                data["vulnerability_explanation"] = gt_dict["explanation"]

        # If ground_truth was missing, construct it from root fields
        if not data.get("ground_truth"):
            data["ground_truth"] = {
                "threats": [data.get("threat_description")] if data.get("threat_description") else [],
                "cwe": data.get("cwe", []),
                "owasp": data.get("owasp_category", []),
                "security_requirements": data.get("security_requirements", []),
                "security_strategy": data.get("security_strategy", []),
                "capability_plan": data.get("capability_plan", []),
                "recommended_capabilities": data.get("capability_plan", []),
                "agent_plan": data.get("agent_plan", []),
                "explanation": data.get("vulnerability_explanation"),
                "repair_code": data.get("secure_repair"),
                "verification": data.get("verification"),
            }

        return data

    def to_scf_payload(self) -> Dict[str, Any]:
        """Converts to dictionary compatible with SecurityContextPayload."""
        target_dict = self.ground_truth.model_dump(exclude_none=True) if self.ground_truth else {}
        return {
            "task": self.task_type.value,
            "requirement": self.requirement,
            "prompt": self.developer_prompt or self.prompt,
            "language": self.language or self.context.get("language"),
            "framework": self.framework or self.context.get("framework"),
            "code": self.source_code or self.code,
            "security_findings": [f.model_dump() for f in self.security_findings],
            "project_metadata": self.project_context or self.context,
            "target_output": target_dict,
        }
