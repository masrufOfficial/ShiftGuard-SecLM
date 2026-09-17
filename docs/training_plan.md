# Training Plan: ShiftGuard-SecLM

## 1. Multi-Stage Training Protocol

Training proceeds in four distinct stages, beginning strictly with random parameter initialization:

```text
Stage 0: Tokenizer Build & Alignment Check
                    │
                    ▼
Stage 1: Domain Pretraining (Causal LM on Code + Security Corpus, 2.5B tokens)
                    │
                    ▼
Stage 2: Multi-Task Security Supervised Training (Tasks 1-11, 500M tokens)
                    │
                    ▼
Stage 3: ShiftGuard Contextual & Agent Specialization (200M tokens)
                    │
                    ▼
Stage 4: (Optional) Security DPO / Alignment
```

---

## 2. Stage Details

### Stage 1: Domain Pretraining
- **Target**: 2.5 Billion tokens.
- **Objective**: Standard causal language modeling loss:
  $$\mathcal{L}_{\text{pretrain}} = -\sum_{t=1}^T \log P(w_t \mid w_{<t})$$
- **Hyperparameters**:
  - Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$, weight decay = $0.1$).
  - Peak Learning Rate: $3.0 \times 10^{-4}$ with 2,000 warm-up steps followed by cosine annealing decay down to $3.0 \times 10^{-5}$.
  - Sequence Length: 2,048 (scaled to 4,096 in final 20% of steps).
  - Global Batch Size: 256 sequences ($\approx 524,288$ tokens per optimizer step).
  - Gradient Checkpointing: Active.
  - Precision: Mixed precision (bfloat16 or fp16).

### Stage 2: Multi-Task Security Supervised Training
- **Target**: 500 Million tokens.
- **Tasks**: Tasks 1 through 11 (Threat Reasoning, CWE, OWASP, Risk, Vulnerability Analysis, Strategy, Agent Plan, Evidence, Repair, Verification).
- **Prompt Loss Masking**: The loss is computed **only on output tokens** (labels for context tokens are set to `-100`).
  $$\mathcal{L}_{\text{multitask}} = -\sum_{t \in \text{OutputTokens}} \log P(w_t \mid w_{<t})$$
- **Task-Balanced Sampling**: Tasks with lower frequency (e.g., re-verification) are up-sampled to maintain an equilibrium across all 11 task categories.

### Stage 3: ShiftGuard Contextual Specialization
- **Target**: 200 Million tokens.
- **Focus**: End-to-end multi-turn security reasoning connecting requirements, static analyzer findings, threat classification, and repair guidance into structured JSON outputs.

---

## 3. Checkpointing & Fault Tolerance
- Checkpoints saved every 1,000 steps with optimizer state, scheduler state, and random seeds.
- Sharded checkpoints supported via `safetensors`.
- Resumable training designed to tolerate Kaggle notebook session timeouts (9-hour / 12-hour limits).

