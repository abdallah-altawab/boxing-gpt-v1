"""
Boxing-GPT — Learning Rate Scheduler
=====================================
Cosine annealing with linear warmup.

This is the standard LR schedule for GPT-style models (used in GPT-2, GPT-3).

Phase 1 — Warmup (0 → warmup_iters):
    LR increases linearly from 0 to max_lr.
    This prevents instability at the very start of training
    when model weights are random and gradients can be huge.

Phase 2 — Cosine Decay (warmup_iters → lr_decay_iters):
    LR decreases following a cosine curve from max_lr → min_lr.
    Cosine decay is smooth and well-behaved.

Phase 3 — Constant (lr_decay_iters → ∞):
    LR stays at min_lr for any steps beyond lr_decay_iters.

                max_lr ─────╮
                           ╰──╮
                              ╰──╮
                                 ╰──╮
                min_lr ────────────────────────────
                0    warm      decay_end     max
"""

import math


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """
    Compute learning rate for a given training step.

    Args:
        step           : current training step (0-indexed)
        max_lr         : peak learning rate
        min_lr         : minimum learning rate floor
        warmup_iters   : number of linear warmup steps
        lr_decay_iters : step at which cosine decay completes

    Returns:
        Current learning rate (float)
    """

    # Phase 1: linear warmup
    if step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters

    # Phase 3: beyond decay point — return min_lr
    if step >= lr_decay_iters:
        return min_lr

    # Phase 2: cosine decay
    # Map step into [0, 1] range within the decay phase
    progress = (step - warmup_iters) / max(1, lr_decay_iters - warmup_iters)

    # Cosine decay: starts at 1.0, ends at 0.0
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    # Scale from min_lr to max_lr
    return min_lr + cosine_decay * (max_lr - min_lr)


def apply_lr(optimizer, lr: float) -> None:
    """Set learning rate for all parameter groups in an optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
