"""Inference script: transcribe audio files using a trained model."""

import argparse
import sys

import soundfile as sf
import torch

from config import AudioConfig, ModelConfig
from dataset import MelSpectrogramTransform, indices_to_text
from model import SpeechToTextModel


def greedy_decode(output: torch.Tensor) -> str:
    """Greedy CTC decoding: take argmax at each timestep, collapse repeats, remove blanks."""
    # output: (time, n_class)
    indices = output.argmax(dim=-1)  # (time,)

    decoded = []
    prev_idx = -1
    for idx in indices.tolist():
        if idx != prev_idx and idx != 0:  # skip repeats and blanks
            decoded.append(idx)
        prev_idx = idx

    return indices_to_text(decoded)


def beam_search_decode(output: torch.Tensor, beam_width: int = 10) -> str:
    """Simple beam search CTC decoding for better accuracy."""
    # output: (time, n_class) log probabilities
    T, C = output.shape

    # Each beam: (sequence, last_char, log_prob)
    beams = [([], -1, 0.0)]

    for t in range(T):
        new_beams = []
        log_probs = output[t]

        for seq, last_char, score in beams:
            # Top-k candidates at this timestep
            topk_probs, topk_indices = log_probs.topk(beam_width)

            for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                new_score = score + prob

                if idx == 0:  # blank
                    new_beams.append((seq, -1, new_score))
                elif idx == last_char:  # repeat -> collapse
                    new_beams.append((seq, last_char, new_score))
                else:
                    new_beams.append((seq + [idx], idx, new_score))

        # Keep top beams
        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

    best_seq = beams[0][0]
    return indices_to_text(best_seq)


def load_model(checkpoint_path: str, device: torch.device) -> SpeechToTextModel:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", ModelConfig())
    model = SpeechToTextModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    loss = checkpoint.get("loss", "?")
    print(f"Loaded model from epoch {epoch} (loss={loss})")

    return model


def transcribe(model: SpeechToTextModel, audio_path: str, device: torch.device,
               use_beam_search: bool = False, beam_width: int = 10) -> str:
    """Transcribe a single audio file."""
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform).transpose(0, 1)

    mel_transform = MelSpectrogramTransform(AudioConfig())
    spectrogram = mel_transform(waveform, sample_rate)  # (time, n_mels)
    spectrogram = spectrogram.unsqueeze(0).to(device)  # (1, time, n_mels)

    with torch.no_grad():
        output = model(spectrogram)  # (1, time, n_class)
        output = output.squeeze(0)  # (time, n_class)

    if use_beam_search:
        return beam_search_decode(output, beam_width)
    return greedy_decode(output)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to text")
    parser.add_argument("audio_path", help="Path to audio file (WAV, FLAC, MP3)")
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--beam-search", action="store_true",
                        help="Use beam search decoding (slower but more accurate)")
    parser.add_argument("--beam-width", type=int, default=10,
                        help="Beam width for beam search")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    try:
        model = load_model(args.checkpoint, device)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Train a model first with: python train.py")
        sys.exit(1)

    text = transcribe(model, args.audio_path, device,
                      use_beam_search=args.beam_search,
                      beam_width=args.beam_width)
    print(f"\nTranscription:\n{text}")


if __name__ == "__main__":
    main()
