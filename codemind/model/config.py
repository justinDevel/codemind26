"""
CodeMind Model Configuration
Novel mini LLM for coding + chat
"""

from dataclasses import dataclass


@dataclass
class CodeMindConfig:
    # Model dimensions
    vocab_size: int = 32000
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    ffn_dim: int = 3072
    max_seq_len: int = 2048

    # Novel: Scope-Relative Position Encoding
    max_scope_depth: int = 16        # max nesting depth tracked
    srpe_dim: int = 64               # scope position embedding size

    # Novel: Syntax-Gated Attention
    use_syntax_gate: bool = True     # enable SGA
    syntax_gate_heads: int = 4       # heads dedicated to syntax gating

    # Novel: Micro Mixture-of-Experts FFN
    use_micro_moe: bool = True
    num_experts: int = 4
    experts_per_token: int = 2       # top-k experts activated

    # Novel: Dual-KV Cache (code vs text streams)
    use_dual_kv: bool = True

    # Training
    dropout: float = 0.1
    attention_dropout: float = 0.05
    layer_norm_eps: float = 1e-6

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    code_start_token_id: int = 3     # <|code|>
    code_end_token_id: int = 4       # <|/code|>
    user_token_id: int = 5           # <|user|>
    assistant_token_id: int = 6      # <|assistant|>
