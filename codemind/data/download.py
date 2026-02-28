"""
Download and prepare public coding datasets for CodeMind training.
Uses HuggingFace datasets library.
"""

import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)

# Datasets to use (small subsets for mini training)
DATASETS = [
    # (hf_path, subset, split, text_field, max_samples)
    ("codeparrot/github-code", "Python", "train", "code", 50000),
    ("code-search-net/code_search_net", "python", "train", "func_code_string", 20000),
    ("sahil2801/CodeAlpaca-20k", None, "train", "output", 20000),
]


def tokenize_and_save(texts: list, out_path: str, vocab_size: int = 32000):
    """
    Simple BPE-like tokenization using sentencepiece if available,
    else character-level. Saves as uint16 binary.
    """
    try:
        import sentencepiece as spm
        # Train a small SP model on the data first if not exists
        sp_model_path = os.path.join(DATA_DIR, 'codemind.model')
        if not os.path.exists(sp_model_path):
            print("Training SentencePiece tokenizer...")
            tmp_txt = os.path.join(DATA_DIR, 'corpus.txt')
            with open(tmp_txt, 'w', encoding='utf-8') as f:
                for t in texts[:10000]:
                    f.write(t[:500] + '\n')
            spm.SentencePieceTrainer.train(
                input=tmp_txt,
                model_prefix=os.path.join(DATA_DIR, 'codemind'),
                vocab_size=vocab_size,
                character_coverage=0.9995,
                model_type='bpe',
                pad_id=0, bos_id=1, eos_id=2, unk_id=3,
                user_defined_symbols=['<|code|>', '<|/code|>', '<|user|>', '<|assistant|>', '<|system|>'],
            )
            print(f"Tokenizer saved to {sp_model_path}")

        sp = spm.SentencePieceProcessor()
        sp.Load(sp_model_path)
        encode = lambda t: sp.EncodeAsIds(t)
    except ImportError:
        print("sentencepiece not found, using char-level tokenization")
        encode = lambda t: [min(ord(c), vocab_size - 1) for c in t]

    all_ids = []
    for text in tqdm(texts, desc="Tokenizing"):
        ids = encode(text) + [2]  # append EOS
        all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(out_path)
    print(f"Saved {len(arr):,} tokens to {out_path}")
    return len(arr)


def main():
    all_texts = []

    for hf_path, subset, split, field, max_samples in DATASETS:
        print(f"\nLoading {hf_path}...")
        try:
            if subset:
                ds = load_dataset(hf_path, subset, split=split, streaming=True)
            else:
                ds = load_dataset(hf_path, split=split, streaming=True)

            count = 0
            for sample in ds:
                text = sample.get(field, '')
                if text and len(text) > 50:
                    all_texts.append(text)
                    count += 1
                    if count >= max_samples:
                        break
            print(f"  Loaded {count} samples")
        except Exception as e:
            print(f"  Failed to load {hf_path}: {e}")

    if not all_texts:
        print("No data loaded. Check your internet connection and dataset names.")
        return

    print(f"\nTotal samples: {len(all_texts):,}")

    # Split train/val
    split_idx = int(len(all_texts) * 0.98)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]

    tokenize_and_save(train_texts, os.path.join(DATA_DIR, 'train.bin'))
    tokenize_and_save(val_texts, os.path.join(DATA_DIR, 'val.bin'))
    print("\nData preparation complete.")


if __name__ == '__main__':
    main()
