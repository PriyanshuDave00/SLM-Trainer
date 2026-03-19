"""
Token-level tokenizer for the language model.

This tokenizer:
1) Splits text into words and punctuation tokens.
2) Builds a fixed vocabulary from the most frequent tokens.
3) Encodes text into token IDs and decodes IDs back to text.
"""

import re
from typing import Dict, List
from collections import Counter


class TokenTokenizer:
    """
    Simple word-level tokenizer with punctuation handling.

    The vocabulary includes:
    - <PAD> for padding
    - <UNK> for out-of-vocabulary tokens
    - The most common tokens from the training corpus
    """

    def __init__(self):
        """Initialize empty vocabularies and reserved tokens."""
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.vocab_size: int = 0

        # Special tokens.
        self.UNK = "<UNK>"
        self.PAD = "<PAD>"

    # -----------------------------
    # Tokenization logic
    # -----------------------------
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens: words, numbers, punctuation.

        Example:
            "Hello, world!" -> ["Hello", ",", "world", "!"]
        """
        return re.findall(r"\w+|[^\w\s]", text)

    # -----------------------------
    # Build vocab
    # -----------------------------
    def build_vocab(self, text: str) -> None:
        """
        Build a fixed vocabulary from the input text.

        The vocab is capped to the most frequent 10k tokens.
        """
        tokens = self.tokenize(text)

        # Count token frequencies for pruning.
        counter = Counter(tokens)

        most_common = [tok for tok, _ in counter.most_common(10000)]

        # Include special tokens at the front of the vocabulary.
        all_tokens = [self.PAD, self.UNK] + most_common

        self.vocab = {tok: i for i, tok in enumerate(all_tokens)}
        self.reverse_vocab = {i: tok for tok, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    # -----------------------------
    # Encode
    # -----------------------------
    def encode(self, text: str) -> List[int]:
        """
        Convert text into a list of token IDs.

        Unknown tokens are mapped to <UNK>.
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(tok, self.vocab[self.UNK]) for tok in tokens]

    # -----------------------------
    # Decode
    # -----------------------------
    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back into a human-readable string.

        This uses a simple spacing rule to avoid spaces before punctuation.
        """
        tokens = [self.reverse_vocab.get(i, self.UNK) for i in ids]

        # Smart join (fix spacing around punctuation).
        text = ""
        for tok in tokens:
            if tok in {".", ",", "!", "?", ":", ";"}:
                text += tok
            elif len(text) == 0:
                text += tok
            else:
                text += " " + tok

        return text

    # -----------------------------
    # Save / Load
    # -----------------------------
    def to_dict(self) -> dict:
        """Serialize tokenizer state to a plain dict."""
        return {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenTokenizer":
        """Reconstruct a TokenTokenizer from a serialized dict."""
        tok = cls()
        tok.vocab = data["vocab"]
        tok.reverse_vocab = {int(v): k for k, v in tok.vocab.items()}
        tok.vocab_size = data["vocab_size"]
        return tok
