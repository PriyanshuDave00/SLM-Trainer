"""
Training utilities for a GPT-style language model.

This module provides:
1) A random batch sampler for token sequences.
2) A warmup + cosine learning-rate scheduler.
3) A training function that builds the tokenizer and model.
"""

from typing import Tuple

import torch
import torch.optim as optim

from model import GPTLanguageModel
from tokenizer import TokenTokenizer


# -----------------------------
# Random Batch Sampler (Scalable)
# -----------------------------
def get_batch(tokens, batch_size, seq_len, device):
    """
    Sample random contiguous sequences for next-token prediction.

    Args:
        tokens: 1D tensor of token IDs.
        batch_size: Number of sequences per batch.
        seq_len: Length of each sequence.
        device: Target device for tensors.

    Returns:
        Tuple of (x, y) where y is x shifted by one token.
    """
    # Random start indices for each sequence in the batch.
    idx = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))

    # Stack sequences into a batch.
    x = torch.stack([tokens[i : i + seq_len] for i in idx])
    y = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in idx])

    return x.to(device), y.to(device)


# -----------------------------
# Warmup + Cosine Scheduler
# -----------------------------
class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay learning-rate schedule."""

    def __init__(self, optimizer, warmup_steps, total_steps):
        """
        Args:
            optimizer: Optimizer whose LR will be updated.
            warmup_steps: Number of warmup steps.
            total_steps: Total number of training steps.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        """Advance the schedule by one step and update optimizer LR."""
        self.step_num += 1

        if self.step_num < self.warmup_steps:
            lr_scale = self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr_scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * lr_scale


# -----------------------------
# Main Training Function
# -----------------------------
def train(
    text,
    epochs=5,
    batch_size=64,
    seq_len=128,
    learning_rate=3e-4,
    embed_dim=256,
    num_heads=4,
    num_layers=6,
    accum_steps=4,
):
    """
    Train a GPTLanguageModel on raw text and return the model + tokenizer.

    Args:
        text: Training corpus as a single string.
        epochs: Number of training epochs.
        batch_size: Batch size for token sequences.
        seq_len: Sequence length for training.
        learning_rate: Initial learning rate for AdamW.
        embed_dim: Model embedding dimension.
        num_heads: Number of attention heads per block.
        num_layers: Number of transformer blocks.
        accum_steps: Gradient accumulation steps.
    """
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Tokenizer ----
    tokenizer = TokenTokenizer()
    tokenizer.build_vocab(text)

    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Dataset tokens: {len(tokens):,}")

    # ---- Model ----
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=embed_dim * 4,
        max_seq_len=seq_len,
        dropout=0.1,
    ).to(device)

    print(f"Model params: {model.num_parameters():,}")

    # ---- Optimizer ----
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Store initial LR for the custom scheduler.
    for group in optimizer.param_groups:
        group["initial_lr"] = learning_rate

    # ---- Training Steps ----
    steps_per_epoch = 2000
    total_steps = steps_per_epoch * epochs

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(0.05 * total_steps),
        total_steps=total_steps,
    )

    # ---- Mixed Precision ----
    scaler = torch.cuda.amp.GradScaler()

    # ---- Training Loop ----
    model.train()
    step = 0

    for epoch in range(epochs):
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            x, y = get_batch(tokens, batch_size, seq_len, device)

            with torch.cuda.amp.autocast():
                _, loss = model(x, y)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            # Apply gradients every accum_steps mini-batches.
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                scheduler.step()

            total_loss += loss.item() * accum_steps
            step += 1

        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model, tokenizer
