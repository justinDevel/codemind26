"""
CodeMind Training Loop
Supports AMD ROCm via PyTorch's HSA backend.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from ..model import CodeMindConfig, CodeMindModel
from .config import TrainConfig


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * cosine


class TextDataset(Dataset):
    """Simple token dataset from pre-tokenized .bin files."""

    def __init__(self, data_path: str, seq_len: int):
        import numpy as np
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = torch.from_numpy(
            self.data[start: start + self.seq_len + 1].astype('int64')
        )
        return chunk[:-1], chunk[1:]  # input, labels


def train(model_cfg: Optional[CodeMindConfig] = None, train_cfg: Optional[TrainConfig] = None):
    model_cfg = model_cfg or CodeMindConfig()
    train_cfg = train_cfg or TrainConfig()

    # Device detection: CUDA → DirectML → CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Training on: CUDA GPU")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            print("Training on: DirectML (AMD GPU via DirectX 12)")
        except ImportError:
            device = torch.device('cpu')
            print("Training on: CPU (slow - consider installing torch-directml)")

    # Model
    model = CodeMindModel(model_cfg).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Optimizer: AdamW with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
        eps=1e-8,
    )

    # Mixed precision: only supported on CUDA, not DirectML or CPU
    is_cuda = device.type == 'cuda'
    use_fp16 = train_cfg.use_fp16 and is_cuda
    if train_cfg.use_fp16 and not is_cuda:
        print("Note: fp16 disabled (only supported on CUDA). Training in fp32.")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # Dataset
    train_path = os.path.join(train_cfg.data_dir, 'train.bin')
    if not os.path.exists(train_path):
        print(f"No data found at {train_path}. Run data/download.py first.")
        return

    dataset = TextDataset(train_path, train_cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    step = 0
    optimizer.zero_grad()
    t0 = time.time()

    for epoch in range(1000):  # effectively infinite, stop at max_steps
        for input_ids, labels in loader:
            if step >= train_cfg.max_steps:
                print("Training complete.")
                return

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Update learning rate
            lr = get_lr(step, train_cfg)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=train_cfg.use_fp16):
                out = model(input_ids=input_ids, labels=labels)
                loss = out['loss'] / train_cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % train_cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % train_cfg.log_every == 0:
                dt = time.time() - t0
                print(f"step {step:6d} | loss {loss.item() * train_cfg.grad_accum_steps:.4f} | lr {lr:.2e} | {dt:.1f}s")
                t0 = time.time()

            if step % train_cfg.save_every == 0 and step > 0:
                ckpt = {
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_cfg': model_cfg,
                    'train_cfg': train_cfg,
                }
                path = os.path.join(train_cfg.checkpoint_dir, f'codemind_step{step}.pt')
                torch.save(ckpt, path)
                print(f"Saved checkpoint: {path}")

            step += 1


if __name__ == '__main__':
    train()
