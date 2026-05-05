"""
Boxing-GPT — Full GPT Language Model
======================================
This is where everything comes together.

Full architecture:
    Input token ids (B, T)
        ↓
    Embeddings (Token + Positional)     → (B, T, d_model)
        ↓
    TransformerBlock × n_layers         → (B, T, d_model)
        ↓
    LayerNorm (final)
        ↓
    Linear head → logits over vocab     → (B, T, vocab_size)
        ↓
    CrossEntropyLoss (during training)

Weight Tying:
    The output linear head (d_model → vocab_size) shares weights
    with the input token embedding (vocab_size → d_model).
    This reduces parameters and is standard in GPT-2.

Parameter count estimate (nano config):
    - Embeddings: vocab_size * d_model + context_len * d_model
    - Per block: ~12 * d_model^2
    - n_layers blocks: n_layers * 12 * d_model^2
    - nano (d_model=384, n_layers=6): ~22M params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .embedding import Embeddings
from .block import TransformerBlock


class BoxingGPT(nn.Module):
    """
    GPT-style autoregressive language model.

    Args:
        vocab_size     : vocabulary size (from your BPE tokenizer)
        context_length : maximum sequence length
        n_layers       : number of transformer blocks
        n_heads        : number of attention heads
        d_model        : embedding / model dimension
        d_ff           : feed-forward hidden dimension
        dropout        : dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_layers = n_layers
        self.d_model = d_model

        # ── 1. Embedding Layer ────────────────────────
        self.embeddings = Embeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            context_length=context_length,
            dropout=dropout,
        )

        # ── 2. Stack of Transformer Blocks ───────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                context_length=context_length,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # ── 3. Final Layer Norm ───────────────────────
        self.final_ln = nn.LayerNorm(d_model)

        # ── 4. Language Modeling Head ─────────────────
        # Projects d_model → vocab_size (produces logits)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ── 5. Weight Tying ───────────────────────────
        # Share weights between token embedding and output head
        # (key trick from GPT-2 paper — reduces parameters)
        self.lm_head.weight = self.embeddings.token_embedding.weight

        # ── 6. Initialize weights ─────────────────────
        self.apply(self._init_weights)
        # Special scaled init for residual projections (GPT-2 trick)
        for name, param in self.named_parameters():
            if name.endswith('out_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        print(f"[BoxingGPT] Initialized | Parameters: {self.count_parameters():,}")

    def _init_weights(self, module: nn.Module) -> None:
        """Standard GPT-2 weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Forward Pass ─────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids : LongTensor (B, T) — token ids
            targets   : LongTensor (B, T) — next-token targets (for loss)
                        If None, only logits are returned (inference mode).

        Returns:
            logits : Tensor (B, T, vocab_size) — unnormalized scores
            loss   : scalar Tensor if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.context_length, (
            f"Input length {T} exceeds context_length {self.context_length}"
        )

        # ── Embed tokens + positions ─────────────────
        x = self.embeddings(input_ids)             # (B, T, d_model)

        # ── Pass through transformer blocks ──────────
        for block in self.blocks:
            x = block(x)                           # (B, T, d_model)

        # ── Final layer norm ──────────────────────────
        x = self.final_ln(x)                       # (B, T, d_model)

        # ── Compute logits ─────────────────────────────
        if targets is not None:
            # Training: compute logits for ALL positions
            logits = self.lm_head(x)               # (B, T, vocab_size)

            # Cross-entropy loss: predict token at position t+1 from position t
            # Reshape: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,                   # ignore padding positions
            )
        else:
            # Inference: only compute logits for the LAST position (efficiency)
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    # ── Autoregressive Generation ─────────────────

    @torch.no_grad()
    # In gpt.py — update the generate() signature:
    def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    repetition_penalty: float = 1.0,   # ADD THIS
) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, and nucleus sampling.

        Args:
            input_ids      : LongTensor (1, T) — prompt token ids
            max_new_tokens : how many new tokens to generate
            temperature    : >1 = more random, <1 = more deterministic, 0 = greedy
            top_k          : keep only top-k logits before sampling
            top_p          : nucleus sampling (keep tokens with cumulative prob ≤ p)
            eos_token_id   : stop generation when this token is produced

        Returns:
            LongTensor (1, T + max_new_tokens)
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            ctx = generated if generated.shape[1] <= self.context_length else \
                  generated[:, -self.context_length:]

            # Forward pass (inference mode — only last position logits)
            logits, _ = self(ctx)                  # (1, 1, vocab_size)
            logits = logits[:, -1, :]              # (1, vocab_size)

            # ── Temperature scaling ────────────────────
            if temperature == 0.0:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                continue

            logits = logits / temperature

            # ── Repetition Penalty ────────────────────
            # For any token that already appears in the context, divide its logit
            # by the penalty factor (>1.0 discourages repetition).
            # Why: at low temperatures the model collapses to high-probability loops.
            # How: if logit > 0, divide (reduce); if logit < 0, multiply (push more negative).
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            
            # ── Top-K filtering ───────────────────────
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                top_k_values, _ = torch.topk(logits, k)
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, float('-inf'))

            # ── Nucleus (Top-P) filtering ──────────────
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative prob above the threshold
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float('-inf')

                # Scatter back to original indexing
                logits = torch.zeros_like(logits).scatter_(
                    1, sorted_indices, sorted_logits
                )

            # ── Sample from distribution ──────────────
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated

    # ── Save / Load ───────────────────────────────

    def save(self, path: str) -> None:
        """Save model state dict."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'context_length': self.context_length,
                'n_layers': self.n_layers,
                'n_heads': self.blocks[0].attention.n_heads,
                'd_model': self.d_model,
                'd_ff': self.blocks[0].ffn.net[0].out_features,
            }
        }, path)
        print(f"[BoxingGPT] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'BoxingGPT':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"[BoxingGPT] Loaded from {path}")
        return model
