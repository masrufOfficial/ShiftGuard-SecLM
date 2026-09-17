# ShiftGuard-SecLM

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Pilot-orange.svg)]()

> **Security-Specialized Language Model for Contextual Shift-Left Security Reasoning in AI-Assisted Software Development**

ShiftGuard-SecLM is an MSc-level research language model trained strictly **from scratch with random weight initialization**. It is specifically designed to perform proactive, contextual security reasoning across software requirements, developer prompts, multi-language source code, and security tool evidence (SAST/DAST/SCA).

ShiftGuard-SecLM serves as the primary **text + code security intelligence component** intended to guide the future ShiftGuard multimodal, multi-agent Shift-Left security framework.

---

## 🎯 Core Research Objective

Investigate whether a decoder-only language model trained from scratch and specialized for software-security reasoning can learn to perform proactive security reasoning across the software development lifecycle, and whether its structured outputs can effectively guide a Shift-Left security architecture.

```text
Requirement / Prompt
        ↓
Security Understanding
        ↓
Threat Reasoning
        ↓
CWE / OWASP Mapping
        ↓
Risk Assessment
        ↓
Security Requirements
        ↓
Security Strategy
        ↓
Security Capability / Agent Plan
        ↓
Evidence Interpretation
        ↓
Repair Guidance & Verification
```

---

## 🏗️ Architecture Overview

- **Backbone**: Decoder-only Causal Transformer trained from scratch.
- **Normalization**: Pre-RMSNorm ($\epsilon = 10^{-5}$) for gradient stability without additive bias.
- **Positioning**: Rotary Position Embeddings (RoPE) applied to query and key projections.
- **Attention**: Causal Scaled Dot-Product Attention (SDPA) with Grouped-Query Attention (GQA) support.
- **FFN**: SwiGLU non-linear feed-forward network ($d_{\text{ffn}} = \frac{8}{3} d_{\text{model}}$).
- **Security Context Fusion (SCF)**: Dedicated structured conditioning mechanism binding natural-language requirements, source code, metadata, and security findings into canonical representations.
- **Output**: Machine-readable JSON schema with strict validation and natural-language explanations.

### Model Scaling Hierarchy

| Configuration | Vocab | Hidden ($d$) | Layers ($L$) | Heads ($H$) | $d_{\text{ffn}}$ | Weight Tying | Parameters | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny / Smoke-Test** | 16,384 | 256 | 6 | 8 | 680 | Tied | **8.9M** | Local CPU tests & CI |
| **Small / Pilot** | 32,000 | 768 | 12 | 12 | 2048 | Tied | **109.5M** | Fast Kaggle pilot runs |
| **Research Target A** | 32,000 | 1024 | 24 | 16 | 2816 | Tied | **341.1M** | **Primary Research Model** |
| **Research Target B** | 32,000 | 1024 | 24 | 16 | 2816 | Untied | **373.9M** | Untied head ablation |
| **Research Target C** | 32,000 | 1280 | 24 | 16 | 3584 | Tied | **528.6M** | Upper scaling variant |

---

## 🚀 Research & Development Workflow

```text
Local VS Code Environment
   │  Architecture + Data Pipeline + Configs + Unit Tests
   ▼
GitHub Repository
   │  Source code, tokenizer configs, dataset manifests, reproducible scripts
   ▼
Kaggle / Cloud GPU Environment (2x T4 or A100)
   │  From-scratch training, validation, multi-task specialization, benchmarking
   ▼
Hugging Face Hub
   │  Model weights, tokenizer, model card, reproduction artifacts
```

---

## 📂 Repository Structure

```text
ShiftGuard-SecLM/
├── configs/            # YAML configurations (model, data, training, evaluation)
├── docs/               # Research specifications, architecture, and compute plans
├── requirements/       # Python dependencies (base, train, eval, dev)
├── src/
│   ├── model/          # Scratch Transformer: RMSNorm, RoPE, SwiGLU, SDPA, SCF
│   ├── tokenizer/      # Custom BPE tokenizer pipeline & domain vocabulary
│   ├── data/           # Ingestion, deduplication, and leakage-safe splitting
│   ├── training/       # Training engine, mixed precision, and checkpointing
│   ├── evaluation/     # CWE, OWASP, Risk, Repair, and Generalization benchmarks
│   ├── inference/      # Constrained JSON generator and schema validation
│   ├── serving/        # FastAPI /analyze API contract
│   └── utils/          # Compute estimations, seeding, logging
├── datasets/           # Manifests, schemas, and processed samples
├── scripts/            # CLI utilities (tokenizer, data prep, training, eval)
├── kaggle/             # Standalone training & evaluation notebooks
└── tests/              # Unit and integration test suite
```

---

## 📜 Citation

If you utilize ShiftGuard-SecLM in your research, please cite:

```bibtex
@misc{shiftguard_seclm_2026,
  author = {BUBT MSc Research Team},
  title = {ShiftGuard-SecLM: Security-Specialized Language Model for Contextual Shift-Left Security Reasoning in AI-Assisted Software Development},
  year = {2026},
  url = {https://github.com/ShiftGuard-SecLM/ShiftGuard-SecLM}
}
```

## ⚖️ License

ShiftGuard-SecLM source code is licensed under the [Apache License 2.0](LICENSE).

