"""
CodeMind Transformer Block and Full Model
Assembles: SGA + SRPE + Micro-MoE into a full decoder-only LLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .config import CodeMindConfig
from .attention import SyntaxGatedAttention, ScopeRelativePositionEncoding
from .moe import MicroMoE


class RMSNorm(nn.Module):
    """RMSNorm - more stable than LayerNorm for LLMs."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class CodeMindBlock(nn.Module):
    """
    Single transformer block with all novel techniques:
    - SyntaxGatedAttention (SGA)
    - Micro-MoE FFN
    - Pre-norm with RMSNorm
    """

    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.layer_norm_eps)

        self.attention = SyntaxGatedAttention(config)

        if config.use_micro_moe:
            self.ffn = MicroMoE(
                config.hidden_dim,
                config.ffn_dim,
                config.num_experts,
                config.experts_per_token,
                config.dropout,
            )
            self.use_moe = True
        else:
            # Fallback standard FFN
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_dim, config.ffn_dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.ffn_dim, config.hidden_dim, bias=False),
                nn.Dropout(config.dropout),
            )
            self.use_moe = False

    def forward(
        self,
        x: torch.Tensor,
        is_code_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple, torch.Tensor]:
        # Attention with residual
        normed = self.attn_norm(x)
        attn_out, present_kv = self.attention(
            normed, is_code_mask, attention_mask, past_kv
        )
        x = x + attn_out

        # FFN with residual
        normed = self.ffn_norm(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(normed)
        else:
            ffn_out = self.ffn(normed)
            aux_loss = torch.tensor(0.0, device=x.device)

        x = x + ffn_out
        return x, present_kv, aux_loss


class CodeMindModel(nn.Module):
    """
    Full CodeMind LLM
    
    Novel architecture summary:
    1. AST-aware token embeddings (code tokens get structural bias)
    2. Scope-Relative Position Encoding (SRPE)
    3. Syntax-Gated Attention (SGA) in every layer
    4. Dual-KV Cache (separate code/text key-value memories)
    5. Micro-MoE FFN (4 experts, top-2 routing per token)
    """

    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)

        # Novel: code token type embedding (adds structural bias for code tokens)
        self.token_type_emb = nn.Embedding(2, config.hidden_dim)  # 0=text, 1=code

        # Novel: Scope-Relative Position Encoding
        self.srpe = ScopeRelativePositionEncoding(
            config.hidden_dim,
            config.max_seq_len,
            config.max_scope_depth,
            config.srpe_dim,
        )

        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CodeMindBlock(config) for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.hidden_dim, config.layer_norm_eps)

        # Language model head (tied weights with embeddings)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, T: int, S: int, device: torch.device) -> torch.Tensor:
        """Causal attention mask."""
        mask = torch.full((T, S), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=S - T + 1)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, S]

    def forward(
        self,
        input_ids: torch.Tensor,                          # [B, T]
        is_code_mask: Optional[torch.Tensor] = None,      # [B, T] float
        scope_positions: Optional[torch.Tensor] = None,   # [B, T] int
        scope_depths: Optional[torch.Tensor] = None,      # [B, T] int
        past_kvs: Optional[List[Tuple]] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        # Default masks if not provided
        if is_code_mask is None:
            is_code_mask = torch.zeros(B, T, device=device)
        if scope_positions is None:
            scope_positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            scope_positions = scope_positions.clamp(0, self.config.max_seq_len - 1)
        if scope_depths is None:
            scope_depths = torch.zeros(B, T, dtype=torch.long, device=device)

        # Embeddings
        tok_emb = self.token_emb(input_ids)
        type_ids = is_code_mask.long()
        type_emb = self.token_type_emb(type_ids)
        x = tok_emb + type_emb

        # Scope-relative position encoding
        x = self.srpe(x, scope_positions.clamp(0, self.config.max_seq_len - 1), scope_depths.clamp(0, self.config.max_scope_depth - 1))
        x = self.emb_dropout(x)

        # Causal mask
        past_len = past_kvs[0][0].shape[2] if past_kvs else 0
        causal_mask = self._make_causal_mask(T, T + past_len, device)

        # Transformer blocks
        present_kvs = []
        total_aux_loss = torch.tensor(0.0, device=device)

        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs else None
            x, present_kv, aux_loss = block(x, is_code_mask, causal_mask, past_kv)
            present_kvs.append(present_kv)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            loss = ce_loss + total_aux_loss / len(self.blocks)

        return {
            "logits": logits,
            "loss": loss,
            "past_kvs": present_kvs,
            "aux_loss": total_aux_loss,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
