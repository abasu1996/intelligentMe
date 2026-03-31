"""Dataset loading and audio preprocessing for speech-to-text."""

import torch
import torchaudio
from torch.utils.data import DataLoader

from config import AudioConfig


# Character mapping: blank=0, then a-z, space, apostrophe
CHARS = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz") + [" ", "'"]
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}


def text_to_indices(text: str) -> list[int]:
    """Convert transcript text to list of integer indices."""
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


def indices_to_text(indices: list[int]) -> str:
    """Convert integer indices back to text, skipping blanks."""
    return "".join(IDX_TO_CHAR[i] for i in indices if i != 0)


class MelSpectrogramTransform:
    """Converts raw waveform to log mel spectrogram features."""

    def __init__(self, config: AudioConfig = AudioConfig()):
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
        )
        self.target_sample_rate = config.sample_rate

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        # waveform: (channels, time) -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        spec = self.transform(waveform)  # (1, n_mels, time)
        spec = (spec + 1e-9).log2()  # log mel spectrogram

        # Normalize per sample
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-9)

        return spec.squeeze(0).transpose(0, 1)  # (time, n_mels)


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform, sample_rate, transcript, *_ in batch:
        spec = mel_transform(waveform, sample_rate)
        spectrograms.append(spec)
        label = torch.IntTensor(text_to_indices(transcript))
        labels.append(label)
        input_lengths.append(spec.shape[0])
        label_lengths.append(len(label))

    # Pad spectrograms to max length in batch
    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    labels = torch.cat(labels)
    input_lengths = torch.IntTensor(input_lengths)
    label_lengths = torch.IntTensor(label_lengths)

    return spectrograms, labels, input_lengths, label_lengths


# Global transform instance used by collate_fn
mel_transform = MelSpectrogramTransform()


def get_dataloader(dataset_type: str = "dev-clean", data_dir: str = "./data",
                   batch_size: int = 16, num_workers: int = 4,
                   shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for LibriSpeech dataset."""
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=data_dir,
        url=dataset_type,
        download=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
