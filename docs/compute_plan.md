# Compute & Hardware Plan: ShiftGuard-SecLM

## 1. Local Hardware Profile
- **Host CPU**: AMD Ryzen 5 7600 (6 Cores / 12 Logical Processors)
- **Host RAM**: 16 GB DDR5
- **Local GPU**: Integrated AMD Radeon Graphics (512 MB shared VRAM; **No NVIDIA CUDA GPU detected**)
- **Local Storage**: Drive `I:` (14.70 GB free), Drive `C:` (47.45 GB free)
- **Local Role**: Code development, unit testing, schema validation, tokenizer training, and tiny-model smoke testing (8.9M params on CPU).

---

## 2. Remote GPU Training Strategy (Kaggle & Cloud)
Full-scale training of the 341M parameter model requires dedicated GPU accelerators:
- **Primary Training Platform**: Kaggle GPU (2x NVIDIA T4, 16 GB VRAM each = 32 GB total).
- **Alternative / Scale Platform**: University / Cloud NVIDIA A100 (40 GB or 80 GB).

### Compute Calculations (341.1M Parameter Model)
$$\text{Total Compute FLOPs} \approx 6 \times P \times N_{\text{tokens}} = 6 \times (341.1 \times 10^6) \times (3.2 \times 10^9) \approx 6.55 \times 10^{18}\text{ FLOPs}$$

| Configuration | Model Params | Tokens | Effective Throughput | Estimated Duration | Platform |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny Smoke Test** | 8.9M | 1M | 1,000 tok/sec (CPU) | **~15 minutes** | Local Ryzen 5 CPU |
| **Small Pilot Run** | 109.5M | 1.0B | ~34 TFLOPs (2x T4) | **~5.4 hours** | Kaggle 2x T4 |
| **Research Model** | 341.1M | 3.2B | ~34 TFLOPs (2x T4) | **~53.5 hours (~2.2 days)** | Kaggle 2x T4 (sharded) |
| **Research Model** | 341.1M | 3.2B | ~140 TFLOPs (1x A100)| **~13.0 hours** | University A100 |

---

## 3. VRAM Budget (341.1M Model on 16 GB GPU)
- **Model Weights (BF16 / FP16)**: $341\text{M} \times 2\text{ bytes} = 682\text{ MB}$
- **Gradients (BF16 / FP16)**: $341\text{M} \times 2\text{ bytes} = 682\text{ MB}$
- **Optimizer States (8-bit AdamW)**: $341\text{M} \times 4\text{ bytes} = 1.36\text{ GB}$
- **Activations (Seq Len = 2048, Batch = 4, with Gradient Checkpointing)**: $\approx 3.5\text{ GB}$
- **Total Peak VRAM**: **~6.5 GB to 8.5 GB** (easily fits within a 16 GB T4 or P100 memory limit).

