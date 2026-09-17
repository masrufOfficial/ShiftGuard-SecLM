# Architecture Specification: ShiftGuard-SecLM

## 1. Architectural Foundations

ShiftGuard-SecLM employs a modern decoder-only causal Transformer architecture engineered for robust numerical stability, efficient memory utilization, and structured generation.

```text
[Input Tokens: Context + Code + Task Token]
                     │
                     ▼
             Embedding Layer (Vocab -> d_model)
                     │
         ┌───────────┴───────────┐
         │  Transformer Block ×L │
         │  ┌─────────────────┐  │
         │  │   Pre-RMSNorm   │  │
         │  └────────┬────────┘  │
         │           ▼           │
         │  ┌─────────────────┐  │
         │  │ Causal SDPA +   │  │
         │  │ RoPE Attention  │  │
         │  └────────┬────────┘  │
         │           ▼           │
         │       Residual +      │
         │  ┌─────────────────┐  │
         │  │   Pre-RMSNorm   │  │
         │  └────────┬────────┘  │
         │           ▼           │
         │  ┌─────────────────┐  │
         │  │   SwiGLU FFN    │  │
         │  └────────┬────────┘  │
         │           ▼           │
         │       Residual +      │
         └───────────┬───────────┘
                     ▼
               Final RMSNorm
                     │
                     ▼
           Tied LM Head / Unembed
                     │
                     ▼
          [Logits over Vocabulary]
```

## 2. Core Mathematical Components

### 2.1 Root Mean Square Normalization (RMSNorm)
RMSNorm replaces traditional LayerNorm by scaling activations without mean-centering:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
where $\gamma \in \mathbb{R}^d$ is a learned gain parameter and $\epsilon = 10^{-5}$. This yields faster computation and improved gradient propagation in mixed precision.

### 2.2 Rotary Position Embedding (RoPE)
RoPE encodes relative position by rotating Query and Key vectors in the complex 2D subspace:
$$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$
where $\theta_i = \theta_0^{-2(i-1)/d}$ with base frequency $\theta_0 = 10,000.0$. RoPE introduces no learned parameters and preserves distance-dependent attention decay.

### 2.3 SwiGLU Feed-Forward Network
The feed-forward sublayer uses the Swish-Gated Linear Unit:
$$\text{SwiGLU}(x) = \left(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right) W_{\text{down}}$$
The hidden expansion dimension is set to $d_{\text{ffn}} = \left\lfloor \frac{8}{3} d_{\text{model}} \right\rfloor$ aligned to multiples of 128 (e.g., $d_{\text{ffn}} = 2816$ for $d = 1024$).

### 2.4 Scaled Dot-Product Attention (SDPA) & Grouped-Query Attention (GQA)
Attention uses causal masking $M_{ij} = -\infty$ for $j > i$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
Implemented through PyTorch's native `torch.nn.functional.scaled_dot_product_attention`, taking advantage of FlashAttention-2 or memory-efficient kernels where available.

For the primary 1B model (`ShiftGuard-SecLM-1B`), **Grouped-Query Attention (GQA)** with an $n_{\text{heads}} : n_{\text{kv\_heads}} = 2:1$ ratio (16 Query heads to 8 Key-Value heads) is deployed. Key-Value projections are repeated across Query head pairs, reducing KV-cache VRAM by 50% during long-context (4K/8K) inference while preserving the full representational power of multi-head attention.

### 2.5 Security Context Fusion (SCF)
SCF serializes heterogeneous security metadata into canonical token sequences delimited by explicit boundary tokens:
```text
<SEC_CONTEXT>
<LANG:python> <FRAMEWORK:fastapi>
<PROMPT:Create bank transfer endpoint>
<CODE:def transfer(...): ...>
<EVIDENCE:semgrep rule="sql-injection" severity="high">
</SEC_CONTEXT>
<TASK:SECURITY_PLAN>
<SCHEMA_START>
{ ... structured JSON output ... }
<SCHEMA_END>
```
During training, cross-entropy loss is computed **strictly on tokens following `<TASK:...>` and `<SCHEMA_START>`**, ensuring the model learns to reason from context rather than simply predicting metadata headers.

---

## 3. Parameter Configurations

| Parameter | Tiny (Smoke) | Small (Pilot) | Research Prototype (V1) | Prototype Untied | Scaled 528M | **ShiftGuard-SecLM-1B (Primary V2)** | **1B Untied Variant** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Vocabulary ($V$) | 16,384 | 32,000 | 32,000 | 32,000 | 32,000 | **32,000** | **32,000** |
| Hidden Dimension ($d$) | 256 | 768 | 1,024 | 1,024 | 1,280 | **2,048** | **2,048** |
| Layers ($L$) | 6 | 12 | 24 | 24 | 24 | **20** | **20** |
| Query Heads ($H_q$) | 8 | 12 | 16 | 16 | 16 | **16** | **16** |
| KV Heads ($H_{kv}$) | 8 | 12 | 16 | 16 | 16 | **8 (GQA 2:1)** | **8 (GQA 2:1)** |
| Head Dimension ($d_k$) | 32 | 64 | 64 | 64 | 80 | **128** | **128** |
| FFN Dimension ($d_{\text{ffn}}$) | 680 | 2,048 | 2,816 | 2,816 | 3,584 | **5,504** | **5,504** |
| Tied Embeddings | Yes | Yes | Yes | No | Yes | **Yes** | **No** |
| Max Context Length | 1,024 | 2,048 | 4,096 | 4,096 | 4,096 | **4,096** | **4,096** |
| **Total Parameter Count** | **8.90M** | **109.53M** | **341.10M** | **373.87M** | **528.61M** | **993.61M** | **1,059.15M** |
| **Role** | Smoke CI | Fast Pilot | Baseline (V1) | Untied Ablation | Scaling Test | **Primary Research Model** | Architecture Ablation |

