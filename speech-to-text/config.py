"""Configuration for the speech-to-text model."""

from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_cnn_layers: int = 3
    n_rnn_layers: int = 5
    rnn_dim: int = 512
    n_class: int = 29  # 26 letters + space + apostrophe + blank (CTC)
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 5.0
    num_workers: int = 0
    dataset_url: str = "dev-clean"  # start small; use "train-clean-100" for real training
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 20
