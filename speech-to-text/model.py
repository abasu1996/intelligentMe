"""Neural network architecture for speech-to-text.

Architecture: CNN feature extractor -> BiLSTM encoder -> FC classifier + CTC loss
"""

import torch
import torch.nn as nn

from config import ModelConfig


class CNNLayerNorm(nn.Module):
    """Layer norm applied per-channel on CNN output."""

    def __init__(self, n_feats: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channel, feature, time)
        x = x.transpose(2, 3)  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3)


class ResidualCNN(nn.Module):
    """Residual CNN block with batch norm and dropout."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int,
                 stride: int, dropout: float, n_feats: int):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        return x + residual


class BiLSTM(nn.Module):
    """Bidirectional LSTM with layer norm."""

    def __init__(self, rnn_dim: int, hidden_size: int, dropout: float, batch_first: bool):
        super().__init__()
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.lstm = nn.LSTM(
            input_size=rnn_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x


class SpeechToTextModel(nn.Module):
    """Full speech-to-text model: CNN + BiLSTM + FC with CTC."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        n_feats = config.n_mels

        # Initial CNN to process spectrogram
        self.cnn = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)

        n_feats = n_feats // 2  # due to stride=2

        # Residual CNN layers
        self.res_cnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=config.dropout, n_feats=n_feats)
            for _ in range(config.n_cnn_layers)
        ])

        self.fc_after_cnn = nn.Linear(n_feats * 32, config.rnn_dim)

        # Bidirectional LSTM layers
        self.bilstm_layers = nn.Sequential(*[
            BiLSTM(
                rnn_dim=config.rnn_dim if i == 0 else config.rnn_dim * 2,
                hidden_size=config.rnn_dim,
                dropout=config.dropout,
                batch_first=True,
            )
            for i in range(config.n_rnn_layers)
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.rnn_dim * 2, config.rnn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.rnn_dim, config.n_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, n_mels) - mel spectrogram input

        Returns:
            (batch, time, n_class) - log probabilities for CTC
        """
        # (batch, time, n_mels) -> (batch, 1, n_mels, time)
        x = x.unsqueeze(1).transpose(2, 3)

        x = self.cnn(x)
        x = self.res_cnn_layers(x)

        # (batch, channels, features, time) -> (batch, time, channels * features)
        batch, channels, features, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * features)

        x = self.fc_after_cnn(x)

        for bilstm in self.bilstm_layers:
            x = bilstm(x)

        x = self.classifier(x)

        return x.log_softmax(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
