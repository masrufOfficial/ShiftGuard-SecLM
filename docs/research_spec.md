# Research Specification: ShiftGuard-SecLM

## 1. Research Identity & Context
- **Project Title**: ShiftGuard-SecLM: Security-Specialized Language Model for Contextual Shift-Left Security Reasoning in AI-Assisted Software Development
- **Principal Researcher / Author**: Masruf Rahman
- **Scope**: Research Initiative $\to$ Conference Submission $\to$ Extended Journal Paper
- **Core Principle**: ShiftGuard-SecLM is trained **strictly from scratch with random weight initialization**. It is neither a fine-tuned checkpoint of an existing LLM (e.g., Llama, Qwen, DeepSeek, CodeLlama) nor an API prompt wrapper.

## 2. Primary Research Question (PRQ)
> *Can a compact decoder-only language model (~350M–500M parameters) trained from scratch on a domain-specialized security and code corpus learn proactive, structured security reasoning across software requirements, source code, and security evidence, and how effectively can its structured representations guide an automated Shift-Left security architecture compared to general-purpose code models and heuristic baselines?*

## 3. Sub-Research Questions (SRQs)
- **SRQ 1 (Contextual Fusion)**: How effectively does Security Context Fusion (SCF) synthesize disparate modalities (natural-language requirement, AST/code patterns, static analyzer outputs) into unified threat models?
- **SRQ 2 (Taxonomy Mapping)**: What accuracy and macro-F1 does the model achieve when mapping code snippets and prompt vulnerabilities to fine-grained MITRE CWE and OWASP Top 10 categories?
- **SRQ 3 (Calibration & Risk Assessment)**: Are the model's confidence scores well-calibrated (measured via Expected Calibration Error and Brier Score) across diverse severity classes?
- **SRQ 4 (Automated Repair & Verification)**: Can the model generate syntactically and semantically sound security patches that pass functional unit tests while eliminating reported static analysis findings without regressions?
- **SRQ 5 (Out-of-Distribution Generalization)**: How does the model generalize to unseen application domains, unfamiliar third-party libraries, and AI-generated code?

## 4. Operational Boundaries & Scope
- **Current Phase**: **Model Only**. The model learns the representations, reasoning chains, structured output schemas, and inference APIs.
- **Future Integration**: The full ShiftGuard framework (multi-agent orchestration, blockchain provenance, VS Code plugin, dynamic DAST sandbox) will interface with ShiftGuard-SecLM via the standardized API contract (`POST /analyze`).

