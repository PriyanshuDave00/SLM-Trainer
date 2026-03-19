"""
Training entrypoint for building a language model from a text or PDF file.

This module:
1) Loads a source document from disk.
2) Trains a GPT-style model using the shared trainer.
3) Saves model weights, tokenizer, and a minimal config snapshot.
"""

import pickle
from pathlib import Path

import torch

from trainer import train


# -----------------------------
# Load file (txt or pdf)
# -----------------------------
def load_file(path: str) -> str:
    """
    Read text from a .txt or .pdf file and return it as a single string.

    Args:
        path: File path to a .txt or .pdf file.

    Returns:
        The extracted text content.
    """
    path = Path(path)

    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8")

    if path.suffix == ".pdf":
        try:
            import PyPDF2
        except ImportError as exc:
            raise ImportError("Install PyPDF2: pip install PyPDF2") from exc

        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                # Some PDF pages may be image-only and return None.
                t = page.extract_text()
                if t:
                    text += t + "\n"

        return text

    raise ValueError("Only .txt and .pdf files are supported.")


# -----------------------------
# Save model + tokenizer
# -----------------------------
def save_model(model, tokenizer, save_dir="story_model"):
    """
    Persist model weights, tokenizer, and a minimal config dict.

    Args:
        model: Trained GPTLanguageModel instance.
        tokenizer: TokenTokenizer instance used during training.
        save_dir: Destination folder for saved artifacts.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Save model weights.
    torch.save(model.state_dict(), save_path / "model.pt")

    # Save tokenizer via pickle for later inference.
    with open(save_path / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Save minimal config for reconstruction.
    config = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": model.embed_dim,
        "num_heads": model.blocks[0].attn.num_heads,
        "num_layers": len(model.blocks),
        "ff_dim": model.blocks[0].ff.net[0].out_features,
        "max_seq_len": model.max_seq_len,
    }

    with open(save_path / "config.pkl", "wb") as f:
        pickle.dump(config, f)

    print(f"\n[INFO] Model saved to '{save_dir}/'")


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    """Interactive CLI for training and saving a model from a file."""
    file_path = input("Enter file path (.txt or .pdf): ").strip()

    print("\n[INFO] Loading file...")
    text = load_file(file_path)

    if len(text) < 500:
        raise ValueError("Text too short. Use a larger file.")

    print(f"[INFO] Loaded {len(text):,} characters")

    # Train with the shared trainer to avoid duplication.
    model, tokenizer = train(
        text=text,
        epochs=10,
        batch_size=32,
        seq_len=128,
        learning_rate=3e-4,  # slightly higher for smaller model
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        accum_steps=2,
    )

    # Save everything to disk.
    save_model(model, tokenizer)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
