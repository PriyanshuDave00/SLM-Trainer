"""
Interactive prompt runner for a trained GPTLanguageModel.

This script loads a saved model and tokenizer, then lets you
enter prompts in a loop to generate text samples.
"""

import pickle
from pathlib import Path

import torch

from model import GPTLanguageModel


# -----------------------------
# Load model + tokenizer
# -----------------------------
def load_model_from_folder(device="cpu"):
    """
    Load a model and tokenizer from a saved folder.

    The folder must contain:
    - model.pt
    - tokenizer.pkl
    """
    folder_path = input("Enter path to trained model folder: ").strip()
    model_dir = Path(folder_path)

    if not model_dir.exists():
        raise ValueError("Folder does not exist.")

    required_files = ["model.pt", "tokenizer.pkl"]
    for filename in required_files:
        if not (model_dir / filename).exists():
            raise ValueError(f"Missing file: {filename}")

    print("[INFO] Loading tokenizer...")
    with open(model_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    print("[INFO] Rebuilding model with fixed hyperparameters...")

    # NOTE: These values must match the training configuration.
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        ff_dim=256 * 4,
        max_seq_len=256,
        dropout=0.0,  # no dropout during inference
    )

    print("[INFO] Loading weights...")
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))

    # Move to device and switch to eval mode for inference.
    model.to(device)
    model.eval()

    print("[INFO] Model loaded successfully.\n")

    return model, tokenizer


# -----------------------------
# Generate text
# -----------------------------
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    device="cpu",
):
    """
    Generate text by sampling from the model one token at a time.

    Args:
        model: Trained GPTLanguageModel.
        tokenizer: TokenTokenizer with the training vocab.
        prompt: Input prompt string.
        max_new_tokens: Number of tokens to generate.
        temperature: Softmax temperature for sampling.
        device: Target device for tensors.
    """
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Use the model's trained context window for conditioning.
    max_len = model.max_seq_len

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to the maximum context window.
            x_cond = x[:, -max_len:]

            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            probs = torch.softmax(logits, dim=-1)

            # Sample the next token from the distribution.
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the running sequence.
            x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


# -----------------------------
# Main loop
# -----------------------------
def main():
    """Interactive loop for generating text from user prompts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model_from_folder(device=device)

    while True:
        prompt = input("Prompt (or 'exit'): ")

        if prompt.lower() == "exit":
            break

        output = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=200,
            temperature=0.8,
            device=device,
        )

        print("\nOutput:\n")
        print(output)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
