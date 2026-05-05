"""
Boxing-GPT — Multi-Head Causal Self-Attention
==============================================
This is the CORE operation of any GPT-style model.

What self-attention does:
    For every token position, it computes a weighted sum of ALL other
    token vectors, where the weights reflect "how relevant is token j
    to understanding token i?".

    The "causal" part: token i can ONLY attend to tokens 0..i
    (no peeking at future tokens). This is enforced by a mask.

Math:
    Q = X @ W_Q     # Queries   (B, T, d_model)
    K = X @ W_K     # Keys      (B, T, d_model)
    V = X @ W_V     # Values    (B, T, d_model)

    # Split into n_heads heads, each of size head_dim = d_model // n_heads
    # Scaled dot-product attention per head:
    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(head_dim)) @ V

    # Mask future positions to -inf before softmax
    # Concatenate all heads, project back: output = concat(heads) @ W_O

Multi-Head:
    Run h independent attention "heads" in parallel,
    each with smaller dimension (d_model / h).
    Then concatenate and project → same d_model dimension.
    Multiple heads = multiple "aspects" of attention simultaneously.

Shape flow:
    Input  : (B, T, d_model)
    Output : (B, T, d_model)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-Head Causal (Masked) Self-Attention.

    Args:
        d_model    : total embedding dimension
        n_heads    : number of attention heads
        context_length : max sequence length (for building the causal mask)
        dropout    : attention dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads        # Dimension per head

        # ── Linear projections ───────────────────────
        # W_Q, W_K, W_V packed into one matrix for efficiency
        # Input: (B, T, d_model) → Output: (B, T, 3 * d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection W_O: maps concatenated heads back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout on attention weights and on output
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # ── Causal mask ──────────────────────────────
        # A lower-triangular matrix of 1s: position i can only see 0..i
        # Registered as a buffer (not a parameter, won't be updated by optimizer)
        # Shape: (1, 1, context_length, context_length) for broadcasting
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer('mask', mask.view(1, 1, context_length, context_length))

        # ── Weight initialization ─────────────────────
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        # Scale output projection by 1/sqrt(n_layers) for stability (GPT-2 trick)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (B, T, d_model)

        Returns:
            Tensor of shape (B, T, d_model)
        """
        B, T, C = x.shape   # C == d_model

        # ── Step 1: Compute Q, K, V ─────────────────
        # Project x → Q, K, V all at once
        qkv = self.qkv_proj(x)                          # (B, T, 3*d_model)
        Q, K, V = qkv.split(self.d_model, dim=-1)       # each (B, T, d_model)

        # ── Step 2: Reshape for multi-head ──────────
        # (B, T, d_model) → (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)   # (B, n_heads, T, head_dim)
        K = split_heads(K)   # (B, n_heads, T, head_dim)
        V = split_heads(V)   # (B, n_heads, T, head_dim)

        # ── Step 3: Scaled dot-product attention ────
        # scores[b, h, i, j] = how much position i attends to position j
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        # ── Step 4: Apply causal mask ───────────────
        # Mask out upper triangle (future tokens) with -inf
        # After softmax, -inf → 0 (no attention to future)
        causal_mask = self.mask[:, :, :T, :T]           # slice to actual T
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # ── Step 5: Softmax → attention weights ─────
        attn_weights = F.softmax(scores, dim=-1)         # (B, n_heads, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        # ── Step 6: Weighted sum of values ──────────
        out = torch.matmul(attn_weights, V)              # (B, n_heads, T, head_dim)

        # ── Step 7: Merge heads ──────────────────────
        # (B, n_heads, T, head_dim) → (B, T, n_heads, head_dim) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # ── Step 8: Output projection ────────────────
        out = self.out_proj(out)                         # (B, T, d_model)
        out = self.resid_dropout(out)

        return out
