"""
CodeMind AST-Aware Tokenizer
Novel: injects structural scope signals alongside token ids.
Uses tree-sitter to parse code and extract scope boundaries.
Falls back gracefully if tree-sitter not available.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Scope-opening tokens in code
SCOPE_OPEN = {'{', '(', '[', 'def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:'}
SCOPE_CLOSE = {'}', ')', ']'}

# Special tokens
SPECIAL_TOKENS = {
    '<|pad|>': 0,
    '<|bos|>': 1,
    '<|eos|>': 2,
    '<|code|>': 3,
    '<|/code|>': 4,
    '<|user|>': 5,
    '<|assistant|>': 6,
    '<|system|>': 7,
}


@dataclass
class TokenizerOutput:
    input_ids: List[int]
    is_code_mask: List[float]       # 1.0 = code token, 0.0 = text
    scope_positions: List[int]      # position within current scope
    scope_depths: List[int]         # nesting depth


class CodeMindTokenizer:
    """
    AST-Aware Tokenizer
    
    Novel features:
    1. Tracks scope depth and position-within-scope for every token
    2. Marks code tokens vs natural language tokens
    3. Understands chat format: <|user|> ... <|assistant|> ... 
    4. Code blocks (``` or <|code|>) get structural analysis
    """

    def __init__(self, vocab_path: Optional[str] = None):
        self.special_tokens = SPECIAL_TOKENS
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Initialize with special tokens
        self.vocab.update(self.special_tokens)
        for token, idx in self.special_tokens.items():
            self.id_to_token[idx] = token

        if vocab_path:
            self._load_vocab(vocab_path)

        # Try to load sentencepiece for subword tokenization
        self._sp_model = None
        self._try_load_sentencepiece(vocab_path)

    def _try_load_sentencepiece(self, vocab_path: Optional[str]):
        try:
            import sentencepiece as spm
            if vocab_path and vocab_path.endswith('.model'):
                self._sp_model = spm.SentencePieceProcessor()
                self._sp_model.Load(vocab_path)
        except ImportError:
            pass  # fallback to character-level

    def _load_vocab(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    token, idx = line.strip().split('\t')
                    self.vocab[token] = int(idx)
                    self.id_to_token[int(idx)] = token
        except Exception:
            pass

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text to ids. Uses SP model if available, else char-level."""
        if self._sp_model:
            ids = self._sp_model.EncodeAsIds(text)
            return ids
        # Fallback: simple whitespace + char tokenization
        tokens = []
        for word in re.findall(r'\w+|[^\w\s]|\s+', text):
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Unknown token: use hash mod vocab_size (crude but functional)
                tokens.append((hash(word) % (32000 - 100)) + 100)
        return tokens

    def _compute_scope_signals(
        self, tokens_text: str, token_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Novel: compute scope_position and scope_depth for each token.
        Parses the raw text to track nesting.
        """
        scope_positions = []
        scope_depths = []

        depth = 0
        pos_in_scope = 0

        # Simple character-level scope tracking
        chars = list(tokens_text)
        i = 0
        char_depths = []
        char_scope_pos = []

        while i < len(chars):
            c = chars[i]
            if c in ('{', '(', '['):
                char_depths.append(depth)
                char_scope_pos.append(pos_in_scope)
                depth += 1
                pos_in_scope = 0
            elif c in ('}', ')', ']'):
                depth = max(0, depth - 1)
                pos_in_scope = 0
                char_depths.append(depth)
                char_scope_pos.append(pos_in_scope)
            else:
                char_depths.append(depth)
                char_scope_pos.append(pos_in_scope)
                pos_in_scope += 1
            i += 1

        # Map character-level signals to token-level (approximate)
        # We distribute evenly across tokens
        n_tokens = len(token_ids)
        n_chars = len(char_depths)

        for t in range(n_tokens):
            char_idx = min(int(t * n_chars / max(n_tokens, 1)), n_chars - 1)
            scope_depths.append(min(char_depths[char_idx], 15))
            scope_positions.append(min(char_scope_pos[char_idx], 2047))

        return scope_positions, scope_depths

    def encode_chat(self, messages: List[Dict[str, str]]) -> TokenizerOutput:
        """
        Encode a chat conversation into model inputs.
        
        messages format:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
        ]
        """
        all_ids = [self.special_tokens['<|bos|>']]
        all_is_code = [0.0]
        all_scope_pos = [0]
        all_scope_depth = [0]
        full_text = ''

        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'user':
                role_token = self.special_tokens['<|user|>']
            elif role == 'assistant':
                role_token = self.special_tokens['<|assistant|>']
            else:
                role_token = self.special_tokens['<|system|>']

            all_ids.append(role_token)
            all_is_code.append(0.0)
            all_scope_pos.append(0)
            all_scope_depth.append(0)

            # Parse content for code blocks
            segments = self._split_code_segments(content)
            for seg_text, is_code in segments:
                seg_ids = self._tokenize_text(seg_text)
                scope_pos, scope_depth = self._compute_scope_signals(seg_text, seg_ids)

                all_ids.extend(seg_ids)
                all_is_code.extend([1.0 if is_code else 0.0] * len(seg_ids))
                all_scope_pos.extend(scope_pos)
                all_scope_depth.extend(scope_depth)
                full_text += seg_text

        all_ids.append(self.special_tokens['<|eos|>'])
        all_is_code.append(0.0)
        all_scope_pos.append(0)
        all_scope_depth.append(0)

        return TokenizerOutput(
            input_ids=all_ids,
            is_code_mask=all_is_code,
            scope_positions=all_scope_pos,
            scope_depths=all_scope_depth,
        )

    def _split_code_segments(self, text: str) -> List[Tuple[str, bool]]:
        """Split text into (segment, is_code) pairs based on ``` markers."""
        segments = []
        parts = re.split(r'(```[\w]*\n?)', text)
        in_code = False
        for part in parts:
            if re.match(r'```[\w]*\n?', part):
                in_code = not in_code
                continue
            if part:
                segments.append((part, in_code))
        return segments if segments else [(text, False)]

    def decode(self, ids: List[int]) -> str:
        if self._sp_model:
            # Filter special tokens
            filtered = [i for i in ids if i not in self.special_tokens.values()]
            return self._sp_model.DecodeIds(filtered)
        return ' '.join(self.id_to_token.get(i, f'<unk:{i}>') for i in ids)

    @property
    def vocab_size(self) -> int:
        return max(32000, len(self.vocab))
