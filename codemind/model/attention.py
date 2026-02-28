"""
Syntax-Gated Attention (SGA) - Novel technique by CodeMind
 
Key idea: attention gates dynamically open/close per head based on
whether the current token is in a code context or natural language context.
Code tokens attend differently to code vs text tokens.
Also implements Dual-KV: separate KV projections for code/text, merged at output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScopeRelativePositionEncoding(nn.Module):
    """
    Novel: Scope-Relative Position Encoding (SRPE)
    
    Instead of only tracking absolute position, we track:
    - Position within current scope (resets at each {, [, (, def, class etc.)
    - Scope depth (nesting level)
    - Absolute position (standard)
    
    This gives the model a natural understanding of code structure.
    """

    def __init__(self, hidden_dim: int, max_seq_len: int, max_scope_depth: int, srpe_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.srpe_dim = srpe_dim

        # Standard absolute position embedding
        self.abs_pos_emb = nn.Embedding(max_seq_len, hidden_dim - srpe_dim)

        # Scope-relative: position within current scope
        self.scope_pos_emb = nn.Embedding(max_seq_len, srpe_dim // 2)

        # Scope depth embedding (how deep in nesting)
        self.scope_depth_emb = nn.Embedding(max_scope_depth, srpe_dim // 2)

        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        scope_positions: torch.Tensor,   # position within current scope [B, T]
        scope_depths: torch.Tensor,       # nesting depth [B, T]
    ) -> torch.Tensor:
        B, T, _ = x.shape

        abs_pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        abs_emb = self.abs_pos_emb(abs_pos)                    # [B, T, H - srpe_dim]
        scope_pos_emb = self.scope_pos_emb(scope_positions)    # [B, T, srpe_dim//2]
        depth_emb = self.scope_depth_emb(scope_depths)         # [B, T, srpe_dim//2]

        # Concatenate all position signals
        full_emb = torch.cat([abs_emb, scope_pos_emb, depth_emb], dim=-1)  # [B, T, H]
        return self.proj(x + full_emb)


class SyntaxGatedAttention(nn.Module):
    """
    Novel: Syntax-Gated Attention (SGA)
    
    Key innovation: a learned gate per attention head that modulates
    attention scores based on token type (code vs text).
    
    - Code-to-code attention: gate opens fully
    - Text-to-text attention: gate opens fully  
    - Code-to-text cross attention: gate is learned/modulated
    - Text-to-code cross attention: gate is learned/modulated
    
    Also: Dual-KV Cache - code tokens and text tokens have separate
    K,V projections. The model learns different "memories" for each domain.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.use_dual_kv = config.use_dual_kv
        self.syntax_gate_heads = config.syntax_gate_heads
        self.scale = math.sqrt(self.head_dim)

        assert self.num_heads * self.head_dim == self.hidden_dim

        # Standard Q projection
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        if self.use_dual_kv:
            # Dual-KV: separate projections for code and text tokens
            self.k_proj_code = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.v_proj_code = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.k_proj_text = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.v_proj_text = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            # Learned blend weight to merge code/text KV
            self.kv_blend = nn.Linear(self.hidden_dim, self.num_heads, bias=True)
        else:
            self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # Syntax gate: produces per-head gate values from token type signal
        # gate_heads get syntax-modulated attention, rest are standard
        self.syntax_gate = nn.Sequential(
            nn.Linear(2, self.syntax_gate_heads * 2, bias=True),  # 2 inputs: is_code_q, is_code_k
            nn.SiLU(),
            nn.Linear(self.syntax_gate_heads * 2, self.syntax_gate_heads, bias=True),
            nn.Sigmoid()
        )

        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

    def forward(
        self,
        x: torch.Tensor,
        is_code_mask: torch.Tensor,       # [B, T] float, 1.0 = code token, 0.0 = text
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        B, T, _ = x.shape

        q = self._split_heads(self.q_proj(x))  # [B, H, T, D]

        if self.use_dual_kv:
            # Compute KV for both code and text streams
            k_code = self._split_heads(self.k_proj_code(x))
            v_code = self._split_heads(self.v_proj_code(x))
            k_text = self._split_heads(self.k_proj_text(x))
            v_text = self._split_heads(self.v_proj_text(x))

            # Blend weight per token per head: how much to use code vs text KV
            blend = torch.sigmoid(self.kv_blend(x))  # [B, T, H]
            blend = blend.transpose(1, 2).unsqueeze(-1)  # [B, H, T, 1]

            k = blend * k_code + (1 - blend) * k_text
            v = blend * v_code + (1 - blend) * v_text
        else:
            k = self._split_heads(self.k_proj(x))
            v = self._split_heads(self.v_proj(x))

        # Append to KV cache if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v)

        S = k.shape[2]  # source sequence length (with cache)

        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, T, S]

        # --- Syntax Gate ---
        # For each query token, compute gate based on (is_code_q, is_code_k)
        # We use mean of is_code_mask over source as a proxy for is_code_k
        is_code_q = is_code_mask.unsqueeze(-1)                          # [B, T, 1]
        is_code_k_mean = is_code_mask.mean(dim=1, keepdim=True).unsqueeze(1)  # [B, 1, 1]
        gate_input = torch.cat([
            is_code_q.expand(B, T, 1),
            is_code_k_mean.expand(B, T, 1)
        ], dim=-1)  # [B, T, 2]

        gate = self.syntax_gate(gate_input)  # [B, T, syntax_gate_heads]
        gate = gate.transpose(1, 2).unsqueeze(-1)  # [B, syntax_gate_heads, T, 1]

        # Apply gate only to syntax_gate_heads, leave rest unchanged
        gated_scores = scores.clone()
        gated_scores[:, :self.syntax_gate_heads, :T, :] = (
            scores[:, :self.syntax_gate_heads, :T, :] * gate
        )

        # Causal mask
        if attention_mask is not None:
            gated_scores = gated_scores + attention_mask

        attn_weights = F.softmax(gated_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        out = self.out_proj(out)

        return out, present_kv
