import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (Shazeer, 2020).
    
    Computes: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(self, d_model: int, d_ffn: int, use_bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=use_bias)
        self.w_up = nn.Linear(d_model, d_ffn, bias=use_bias)
        self.w_down = nn.Linear(d_ffn, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

