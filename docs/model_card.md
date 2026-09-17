# Model Card: ShiftGuard-SecLM

## 1. Model Details
- **Model Name**: ShiftGuard-SecLM
- **Full Title**: Security-Specialized Language Model for Contextual Shift-Left Security Reasoning in AI-Assisted Software Development
- **Architecture**: Decoder-only Causal Transformer (Pre-RMSNorm, RoPE, SwiGLU, Causal SDPA, Tied Embeddings)
- **Parameter Count**: 341.1M (Research Target A) / 8.9M (Tiny Smoke) / 109.5M (Small Pilot)
- **Initialization**: Strictly random weight initialization (trained from scratch)
- **License**: Apache 2.0

## 2. Intended Use
- **Primary Use**: Contextual software security reasoning across requirements, source code, and security tool evidence to assist developers and automated Shift-Left security scanners.
- **Tasks**: Threat reasoning, CWE classification, OWASP Top 10 mapping, risk calibration, security strategy generation, agent planning, security repair guidance, and verification.
- **Out-of-Scope**: General-purpose conversational chatting, binary malware generation, or exploitation against unauthorized third-party systems.

## 3. Training & Technical Specifications
- **Tokenizer**: Custom 32,000-vocabulary BPE with Byte-level fallback and reserved security tokens.
- **Hardware Target**: Trained on Kaggle 2x NVIDIA T4 / University A100 GPU clusters.
- **Context Length**: 2,048 (scaled to 4,096 tokens).

