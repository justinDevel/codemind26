"""Quick start: download data and train CodeMind mini."""
from codemind.data.download import main as download
from codemind.training import train, TrainConfig
from codemind.model import CodeMindConfig

if __name__ == '__main__':
    import sys

    if '--download' in sys.argv:
        download()
    else:
        model_cfg = CodeMindConfig(
            hidden_dim=512,      # smaller for mini
            num_layers=8,
            num_heads=8,
            head_dim=64,
            ffn_dim=2048,
            max_seq_len=1024,
        )
        train_cfg = TrainConfig(
            batch_size=4,
            grad_accum_steps=8,
            max_steps=20000,
            use_fp16=True,
        )
        train(model_cfg, train_cfg)
