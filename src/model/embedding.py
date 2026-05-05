"""
Boxing-GPT — Embedding Layer
============================
Combines Token Embeddings + Positional Embeddings.

Token Embedding:
    Maps each integer token id → a dense vector of size d_model.
    This is just a learned lookup table (nn.Embedding).

Positional Embedding:
    The transformer has no built-in sense of order.
    We add a learned position vector for each position 0..context_length-1.
    GPT-2 uses LEARNED positional embeddings (not sinusoidal).

Final embedding = token_embedding + position_embedding
Then apply dropout.

Shape flow:
    Input  : (B, T)          — batch of B sequences, each of length T
    Output : (B, T, d_model) — each token is now a d_model-dimensional vector
"""

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """
    Token + Positional Embedding layer.

    Args:
        vocab_size     : size of the token vocabulary
        d_model        : embedding dimension (must match transformer d_model)
        context_length : maximum sequence length
        dropout        : dropout probability applied after embedding sum
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_length: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Token embedding table: shape (vocab_size, d_model)
        # Each row is the learned vector for one token
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding table: shape (context_length, d_model)
        # Each row is the learned vector for one position
        self.position_embedding = nn.Embedding(context_length, d_model)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.context_length = context_length

        # ── Weight initialization ─────────────────────
        # Small normal initialization (like GPT-2)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : LongTensor of shape (B, T) — token ids

        Returns:
            Tensor of shape (B, T, d_model) — embedded + positional vectors
        """
        B, T = x.shape
        assert T <= self.context_length, (
            f"Sequence length {T} exceeds context_length {self.context_length}"
        )

        # Token embeddings: (B, T, d_model)
        tok_emb = self.token_embedding(x)

        # Position indices: [0, 1, 2, ..., T-1] — same for every item in batch
        positions = torch.arange(T, device=x.device)          # (T,)
        pos_emb = self.position_embedding(positions)           # (T, d_model)
        # Broadcasting: pos_emb will be added to every item in batch
        pos_emb = pos_emb.unsqueeze(0)                         # (1, T, d_model)

        # Combine and apply dropout
        return self.dropout(tok_emb + pos_emb)
