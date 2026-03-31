"""Training script for the speech-to-text model."""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import AudioConfig, ModelConfig, TrainConfig
from dataset import get_dataloader
from model import SpeechToTextModel


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device, config):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(dataloader):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = output.transpose(0, 1)  # (time, batch, n_class) for CTC

        # Adjust input lengths for CNN stride reduction (stride=2 halves time dim)
        input_lengths = (input_lengths // 2).int()

        loss = criterion(output, labels, input_lengths, label_lengths)

        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1

        if (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
            lr = scheduler.get_last_lr()[0]
            print(f"  Batch {batch_idx + 1} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for spectrograms, labels, input_lengths, label_lengths in dataloader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            output = model(spectrograms).transpose(0, 1)
            input_lengths = (input_lengths // 2).int()

            loss = criterion(output, labels, input_lengths, label_lengths)
            if torch.isfinite(loss):
                total_loss += loss.item()
                num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    train_config = TrainConfig()
    model_config = ModelConfig()

    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Data
    print("Loading dataset (will download on first run)...")
    train_loader = get_dataloader(
        dataset_type=train_config.dataset_url,
        data_dir=train_config.data_dir,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )

    # Model
    model = SpeechToTextModel(model_config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    total_steps = len(train_loader) * train_config.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Training loop
    best_loss = float("inf")
    for epoch in range(1, train_config.epochs + 1):
        start = time.time()
        print(f"\nEpoch {epoch}/{train_config.epochs}")
        print("-" * 40)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, train_config
        )
        elapsed = time.time() - start

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Time: {elapsed:.1f}s")

        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(train_config.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "model_config": model_config,
            }, checkpoint_path)
            print(f"Saved best model (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
