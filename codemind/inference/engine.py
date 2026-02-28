"""
CodeMind Inference Engine
Fast autoregressive generation with KV-cache and streaming.
"""

import torch
import torch.nn.functional as F
from typing import Iterator, List, Optional, Dict

from ..model import CodeMindConfig, CodeMindModel, CodeMindTokenizer


class CodeMindEngine:
    """
    Inference engine with:
    - KV-cache for fast generation
    - Streaming token output
    - Temperature / top-p / top-k sampling
    - Code-aware is_code_mask inference (auto-detects code blocks)
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print(f"Loading model on {self.device}...")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt.get('model_cfg', CodeMindConfig())

        self.model = CodeMindModel(cfg).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        vocab_path = 'data/processed/codemind.model'
        self.tokenizer = CodeMindTokenizer(vocab_path)

        print(f"Model loaded. Parameters: {self.model.count_parameters():,}")

    @torch.no_grad()
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Iterator[str]:
        """Stream generated tokens one by one."""

        # Encode conversation
        enc = self.tokenizer.encode_chat(messages)

        input_ids = torch.tensor([enc.input_ids], dtype=torch.long, device=self.device)
        is_code_mask = torch.tensor([enc.is_code_mask], dtype=torch.float, device=self.device)
        scope_pos = torch.tensor([enc.scope_positions], dtype=torch.long, device=self.device)
        scope_depth = torch.tensor([enc.scope_depths], dtype=torch.long, device=self.device)

        past_kvs = None
        in_code_block = False
        generated_ids = []

        for _ in range(max_new_tokens):
            out = self.model(
                input_ids=input_ids,
                is_code_mask=is_code_mask,
                scope_positions=scope_pos,
                scope_depths=scope_depth,
                past_kvs=past_kvs,
            )

            logits = out['logits'][:, -1, :]  # [1, vocab]
            past_kvs = out['past_kvs']

            # Sampling
            next_id = self._sample(logits, temperature, top_p, top_k)

            if next_id == self.tokenizer.special_tokens['<|eos|>']:
                break

            generated_ids.append(next_id)
            token_text = self.tokenizer.decode([next_id])
            yield token_text

            # Update inputs for next step (single token)
            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)

            # Update code mask for next token
            if '```' in token_text:
                in_code_block = not in_code_block
            is_code_val = 1.0 if in_code_block else 0.0
            is_code_mask = torch.tensor([[is_code_val]], dtype=torch.float, device=self.device)
            scope_pos = torch.tensor([[0]], dtype=torch.long, device=self.device)
            scope_depth = torch.tensor([[0]], dtype=torch.long, device=self.device)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Non-streaming generation."""
        return ''.join(self.generate_stream(messages, **kwargs))

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        if temperature == 0:
            return logits.argmax(dim=-1).item()

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, top_k)[0][:, -1, None]
            logits = logits.masked_fill(logits < kth_val, float('-inf'))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float('-inf')
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
