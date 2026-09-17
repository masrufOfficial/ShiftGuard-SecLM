# Model Card: ShiftGuard-SecLM-1B

## 1. Model Details
- **Model Name**: ShiftGuard-SecLM-1B (Primary Research Model V2)
- **Full Title**: Security-Specialized Language Model for Contextual Shift-Left Security Reasoning in AI-Assisted Software Development
- **Architecture**: Decoder-only Causal Transformer (Pre-RMSNorm, RoPE, SwiGLU, Causal SDPA with 2:1 Grouped-Query Attention, Tied Embeddings)
- **Parameter Count**: 
  - **993.61M** (Primary Research Model: `configs/model/research_1b.yaml`)
  - **1,059.15M** (Untied Embedding Variant: `ShiftGuard-SecLM-1B-Untied`)
  - **341.1M** (Research Prototype Baseline V1: `configs/model/research_341m.yaml`)
  - **109.5M** (Pilot Baseline: `configs/model/pilot_110m.yaml`)
  - **8.9M** (Smoke-Test CI Model: `configs/model/tiny_smoke.yaml`)
- **Author**: Masruf Rahman
- **Initialization**: Strictly random weight initialization (trained from scratch)
- **License**: Apache 2.0

## 2. Intended Use
- **Primary Use**: Contextual software security reasoning across requirements, source code, and security tool evidence to assist developers and automated Shift-Left security scanners.
- **Tasks**: Threat reasoning, CWE classification, OWASP Top 10 mapping, risk calibration, security strategy generation, agent planning, security repair guidance, and verification.
- **Out-of-Scope**: General-purpose conversational chatting, binary malware generation, or exploitation against unauthorized third-party systems.

## 3. Training & Technical Specifications
- **Tokenizer**: Custom 32,000-vocabulary BPE with Byte-level fallback and 34 reserved security tokens.
- **Hardware Target**: Trained on Kaggle 2x NVIDIA T4 / University A100 GPU clusters.
- **Context Length**: 4,096 tokens (scalable to 8,192 with RoPE).
- **Grouped-Query Attention (GQA)**: 16 Query Heads, 8 Key-Value Heads (2:1 ratio) with head dimension 128.

