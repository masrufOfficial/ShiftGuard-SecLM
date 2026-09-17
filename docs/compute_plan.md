# Compute & Hardware Plan: ShiftGuard-SecLM

## 1. Local Hardware Profile
- **Host CPU**: AMD Ryzen 5 7600 (6 Cores / 12 Logical Processors)
- **Host RAM**: 16 GB DDR5
- **Local GPU**: Integrated AMD Radeon Graphics (512 MB shared VRAM; **No NVIDIA CUDA GPU detected**)
- **Local Storage**: Drive `I:` (14.70 GB free), Drive `C:` (47.45 GB free)
- **Local Role**: Code development, unit testing, schema validation, tokenizer training, and tiny-model smoke testing (8.9M params on CPU).

---

## 2. Remote GPU Training Strategy (Kaggle & Cloud)
Full-scale training of the primary **ShiftGuard-SecLM-1B** (~993.6M parameters) and prototype models requires dedicated GPU accelerators:
- **Primary Training Platform**: Kaggle GPU (2x NVIDIA T4, 16 GB VRAM each = 32 GB total).
- **Alternative / Scale Platform**: Cloud / University NVIDIA A100 (40 GB or 80 GB).

### Compute Calculations (Primary 993.61M Parameter Model)
$$\text{Total Compute FLOPs} \approx 6 \times P \times N_{\text{tokens}} = 6 \times (993.61 \times 10^6) \times (8.0 \times 10^9) \approx 4.77 \times 10^{19}\text{ FLOPs (47.7 ExaFLOPs)}$$

| Token Budget | Total FLOPs | Kaggle 2x T4 (DDP, 32GB) | 1x RTX 3090/4090 | 1x NVIDIA A100 (80GB) | 4x A100 Cluster | 8x A100 Cluster |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.0B Tokens (Pilot)** | $5.96 \times 10^{18}$ | 48.7 hours (~2.0 days) | 20.7 hours | 11.8 hours | 3.2 hours | 1.6 hours |
| **4.0B Tokens (Intermediate)** | $2.38 \times 10^{19}$ | 194.8 hours (~8.1 days) | 82.8 hours | 47.3 hours (~2.0 days) | 12.7 hours | 6.4 hours |
| **8.0B Tokens (Primary Target)** | $4.77 \times 10^{19}$ | 389.7 hours (~16.2 days) | 165.6 hours | 94.6 hours (~3.9 days) | 25.5 hours (~1.1 days) | 12.7 hours |
| **20.0B Tokens (Chinchilla)** | $1.19 \times 10^{20}$ | 974.1 hours (~40.6 days) | 414.0 hours | 236.6 hours (~9.9 days) | 63.7 hours (~2.7 days) | 31.8 hours |

---

## 3. VRAM Budget (Primary 1B Model on 16 GB GPU)
With **8-bit AdamW (`bitsandbytes`)** and **Activation Checkpointing (Gradient Checkpointing)** at $S=2048, B=2$:
- **Model Weights (BF16 / FP16)**: $993.6\text{M} \times 2\text{ bytes} = 1.99\text{ GB}$
- **Gradients (BF16 / FP16)**: $993.6\text{M} \times 2\text{ bytes} = 1.99\text{ GB}$
- **Optimizer States (8-bit AdamW)**: $993.6\text{M} \times 4\text{ bytes} = 3.97\text{ GB}$
  *(vs. 15.90 GB with standard FP32 AdamW)*
- **Activations (Seq Len = 2048, Batch = 2, with Checkpointing)**: $\approx 0.38\text{ GB}$
- **Static Working Memory**: **~7.40 GB**
- **Peak Training VRAM**: **~7.78 GB** (comfortably within the 15.0 GB usable limit of a 16 GB Kaggle T4 or P100 GPU).

---

## 4. Kaggle Multi-Session Resumption Protocol
Kaggle enforces a strict 9-hour execution limit per session (30h/week quota). To train the 1B model across sessions:
1. **Automated Epoch Checkpointing**: Save `model.safetensors`, `train_state.pt`, and `config.yaml` after every epoch or $N$ micro-steps.
2. **Kaggle Output Dataset Persistence**: Export the checkpoint directory as a private Kaggle Dataset via the Kaggle API (`kaggle datasets version -m "Epoch N"`).
3. **Seamless Session Handoff**: The subsequent session mounts the previous session's checkpoint dataset and resumes via `--resume-from /kaggle/input/shiftguard-checkpoint/step_XXXXXXX`.
4. **Local / Drive Mirroring**: Checkpoints are pulled to local drive storage `I:` for permanent archiving and evaluation.

