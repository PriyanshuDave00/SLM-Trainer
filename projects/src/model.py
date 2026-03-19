"""
Transformer-based language model built from scratch using PyTorch.

Implements a GPT-style causal language model with:
- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Feed-forward layers
- Layer normalization
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability for attention weights.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for queries, keys, and values.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute self-attention over a batch of sequences.

        Args:
            x: Input tensor of shape (B, T, C).
            mask: Optional causal mask of shape (1, 1, T, T).
        """
        B, T, C = x.shape

        # Project to queries, keys, values, then split into heads.
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention.
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # Mask out future tokens for causal language modeling.
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values.
        out = torch.matmul(attn, V)

        # Recombine heads into (B, T, C).
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Model embedding dimension.
            ff_dim: Hidden dimension in the feed-forward network.
            dropout: Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to the input."""
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer decoder block."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Model embedding dimension.
            num_heads: Number of attention heads.
            ff_dim: Hidden dimension in the feed-forward network.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Apply attention and feed-forward sublayers with residual connections."""
        # Pre-norm attention.
        x = x + self.attn(self.ln1(x), mask)
        # Pre-norm feed-forward.
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    GPT-style causal language model built from scratch.

    This implementation supports character- or token-level training depending on
    the tokenizer used. It uses standard causal masking and a tied input/output
    embedding matrix.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks.
            ff_dim: Hidden dimension in feed-forward layers.
            max_seq_len: Maximum sequence length for positional embeddings.
            dropout: Dropout probability.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token and positional embeddings.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Stack of transformer decoder blocks.
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        # Final normalization and language modeling head.
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie input/output embeddings to reduce parameters.
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize linear and embedding weights with a GPT-like scheme."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask so each position only attends to past positions."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass through the model.

        Args:
            idx: Input token IDs (B, T).
            targets: Optional target token IDs (B, T) for loss computation.

        Returns:
            Tuple of (logits, loss) where loss is None if targets are not provided.
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence too long: {T} > {self.max_seq_len}"

        # Embed tokens and positions, then apply dropout.
        positions = torch.arange(T, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(positions)
        x = self.dropout(tok_emb + pos_emb)

        # Apply masked transformer blocks.
        mask = self._causal_mask(T, idx.device)
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization and projection to vocab.
        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy.
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Auto-regressively generate new tokens from an initial prompt.

        Args:
            idx: Input token IDs (B, T).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Optional top-k filtering for sampling.
        """
        for _ in range(max_new_tokens):
            # Only keep the most recent context window.
            idx_cond = idx[:, -self.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k truncation for better sample quality.
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
