"""
Boxing-GPT — Feed-Forward Network + Transformer Block
=======================================================

FeedForward Network (FFN):
    After attention, each token position is processed INDEPENDENTLY
    through a 2-layer MLP. This is where the model learns to
    "think about" each position individually.

    Architecture:
        Linear(d_model → d_ff)   — expand (d_ff = 4 * d_model usually)
        GELU activation          — smooth non-linearity
        Linear(d_ff → d_model)   — contract back
        Dropout

    GELU (Gaussian Error Linear Unit) is used instead of ReLU
    because it is smoother around 0, which helps GPT-style training.

Transformer Block:
    One full transformer layer = Attention + FFN + two LayerNorms.
    GPT uses "Pre-Norm" (LayerNorm BEFORE each sub-layer),
    which is more stable than the original "Post-Norm" paper.

    Pre-Norm residual structure:
        x = x + Attention(LayerNorm(x))    ← residual connection
        x = x + FFN(LayerNorm(x))          ← residual connection

    Residual connections let gradients flow directly through the
    network during backprop (solve vanishing gradient problem).
"""

import torch
import torch.nn as nn
from .attention import MultiHeadCausalAttention


# ─────────────────────────────────────────────
#  Feed-Forward Network
# ─────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applied to each token position independently and identically.

    Args:
        d_model : input/output dimension
        d_ff    : hidden layer dimension (typically 4 * d_model)
        dropout : dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),                               # smooth nonlinearity
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout),
        )

        # ── Weight initialization ─────────────────────
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor (B, T, d_model)

        Returns:
            Tensor (B, T, d_model)
        """
        return self.net(x)


# ─────────────────────────────────────────────
#  Transformer Block
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    A single GPT-style transformer block.

    Uses Pre-Layer-Normalization (Pre-LN), which is more stable than
    the original post-norm from "Attention Is All You Need".

    Structure:
        x → LayerNorm → MultiHeadCausalAttention → + (residual) → x
        x → LayerNorm → FeedForward               → + (residual) → x

    Args:
        d_model        : embedding dimension
        n_heads        : number of attention heads
        d_ff           : feed-forward hidden dimension
        context_length : max sequence length (for causal mask)
        dropout        : dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        context_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # LayerNorm before attention (Pre-LN)
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-head causal self-attention
        self.attention = MultiHeadCausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            dropout=dropout,
        )

        # LayerNorm before FFN (Pre-LN)
        self.ln2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor (B, T, d_model)

        Returns:
            Tensor (B, T, d_model)
        """
        # Pre-LN attention with residual
        x = x + self.attention(self.ln1(x))

        # Pre-LN FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x
