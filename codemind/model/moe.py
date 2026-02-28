"""
Micro Mixture-of-Experts FFN (Micro-MoE)
Novel: tiny MoE inside every FFN block - 4 experts, activate top-2 per token.
Each expert specializes: code syntax, code logic, natural language, reasoning.
Cheap at 125M scale but gives the model specialization capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """Single FFN expert."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation - better than ReLU for language models
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class MicroMoE(nn.Module):
    """
    Novel: Micro Mixture-of-Experts
    
    4 experts, each ~1/4 the size of a standard FFN.
    Router selects top-2 per token with load balancing loss.
    
    Expert roles (learned, not hardcoded):
    - The model naturally learns to route code syntax to some experts
      and reasoning/NL to others - emergent specialization.
    
    Load balancing: we add auxiliary loss to prevent expert collapse
    (all tokens going to same expert).
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int, experts_per_token: int, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.hidden_dim = hidden_dim

        # Each expert is smaller: ffn_dim // num_experts * 2 to keep param count similar
        expert_ffn_dim = (ffn_dim // num_experts) * 2
        self.experts = nn.ModuleList([
            Expert(hidden_dim, expert_ffn_dim, dropout)
            for _ in range(num_experts)
        ])

        # Router: maps token hidden state to expert scores
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Load balancing loss coefficient
        self.load_balance_coeff = 0.01

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        x_flat = x.view(-1, H)  # [B*T, H]
        N = x_flat.shape[0]

        # Router scores
        router_logits = self.router(x_flat)              # [N, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)  # [N, num_experts]

        # Top-k expert selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.experts_per_token, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # renormalize

        # Compute expert outputs
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            token_mask = (top_k_indices == expert_idx).any(dim=-1)  # [N]
            if not token_mask.any():
                continue

            expert_input = x_flat[token_mask]
            expert_output = self.experts[expert_idx](expert_input)

            # Weight by router probability for this expert
            expert_weights = torch.zeros(N, device=x.device)
            for k in range(self.experts_per_token):
                mask_k = (top_k_indices[:, k] == expert_idx) & token_mask
                expert_weights[mask_k] = top_k_probs[mask_k, k]

            output[token_mask] += expert_output * expert_weights[token_mask].unsqueeze(-1)

        output = output.view(B, T, H)

        # Load balancing auxiliary loss
        # Encourages uniform expert utilization
        expert_usage = router_probs.mean(dim=0)  # [num_experts]
        load_balance_loss = self.load_balance_coeff * (
            self.num_experts * (expert_usage * expert_usage).sum()
        )

        return output, load_balance_loss
