"""Training hyperparameters for CodeMind mini."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/processed"
    val_split: float = 0.02

    # Training
    batch_size: int = 4              # small for 8GB VRAM
    grad_accum_steps: int = 8        # effective batch = 32
    max_steps: int = 50000
    warmup_steps: int = 1000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_seq_len: int = 1024          # shorter for mini training

    # Precision
    use_fp16: bool = True            # essential for 8GB VRAM on ROCm

    # Checkpointing
    save_every: int = 1000
    eval_every: int = 500
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_every: int = 10
    project_name: str = "codemind-mini"
