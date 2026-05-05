"""
Boxing-GPT — Trainer
======================
The training loop that puts everything together.

Key techniques implemented:
  • Gradient accumulation: simulate large batches on small GPU
  • Mixed precision (bfloat16): faster compute, less memory
  • Gradient clipping: prevent exploding gradients
  • Cosine LR schedule with warmup
  • Periodic evaluation on validation set
  • Checkpoint saving / resuming
  • Perplexity logging (exp(loss) — lower is better)
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

from ..model.gpt import BoxingGPT
from .scheduler import get_lr, apply_lr


class Trainer:
    """
    Training engine for BoxingGPT.

    Args:
        model         : BoxingGPT instance
        train_loader  : training DataLoader
        val_loader    : validation DataLoader
        config        : dict of training hyperparameters (from config.yaml)
    """

    def __init__(
        self,
        model: BoxingGPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # ── Device setup ──────────────────────────────
        self.device = torch.device(config.get('device', 'cpu'))
        self.model = self.model.to(self.device)

        # ── Mixed precision ───────────────────────────
        self.mixed_precision = config.get('mixed_precision', False)
        self.dtype = torch.bfloat16 if (
            self.mixed_precision and torch.cuda.is_available() and
            torch.cuda.is_bf16_supported()
        ) else torch.float32

        if self.dtype == torch.bfloat16:
            print("[Trainer] Using bfloat16 mixed precision")

        # ── Optimizer ─────────────────────────────────
        # Use AdamW with weight decay applied ONLY to weight matrices
        # (not to biases or LayerNorm parameters — this is important)
        self.optimizer = self._build_optimizer()

        # ── torch.compile (PyTorch 2.0+) ──────────────
        if config.get('compile', False):
            print("[Trainer] Compiling model with torch.compile() ...")
            self.model = torch.compile(self.model)

        # ── Training state ────────────────────────────
        self.step = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ── Gradient accumulation ──────────────────────
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)

        print(f"[Trainer] Device: {self.device} | dtype: {self.dtype}")
        print(f"[Trainer] Effective batch size: "
              f"{config['batch_size'] * self.grad_accum_steps}")

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        Build AdamW optimizer with weight decay applied selectively.

        Weight decay is a regularization technique that penalizes large weights.
        We apply it to weight matrices but NOT to:
          - bias terms (1D)
          - LayerNorm weights/biases
          - Embedding weights
        """
        # Separate parameters into two groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:  # Weight matrices
                decay_params.append(param)
            else:  # Biases, LayerNorm params, embeddings (1D)
                no_decay_params.append(param)

        optimizer_groups = [
            {'params': decay_params,    'weight_decay': self.config.get('weight_decay', 0.1)},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"[Trainer] Optimizer: {n_decay:,} decay params | {n_no_decay:,} no-decay params")

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.95),   # Standard GPT betas
            eps=1e-8,
        )

    # ── Evaluation ────────────────────────────────

    @torch.no_grad()
    def evaluate(self, eval_iters: int = 100) -> float:
        """
        Compute average validation loss over eval_iters batches.

        Returns:
            Average validation loss (float)
        """
        self.model.eval()
        val_iter = iter(self.val_loader)
        total_loss = 0.0
        count = 0

        for _ in range(eval_iters):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break

            x, y = x.to(self.device), y.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                _, loss = self.model(x, targets=y)

            total_loss += loss.item()
            count += 1

        self.model.train()
        return total_loss / max(count, 1)

    # ── Checkpoint ────────────────────────────────

    def save_checkpoint(self, tag: str = '') -> str:
        """Save model + optimizer + training state."""
        filename = f"checkpoint_step{self.step}{f'_{tag}' if tag else ''}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)

        print(f"[Trainer] Checkpoint saved → {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"[Trainer] Resumed from step {self.step} | best_val_loss={self.best_val_loss:.4f}")

    # ── Main Training Loop ────────────────────────

    def train(
        self,
        max_iters: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        """
        Run the full training loop.

        Args:
            max_iters   : override config max_iters if provided
            resume_from : path to checkpoint to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        max_iters = max_iters or self.config.get('max_iters', 50_000)
        eval_interval    = self.config.get('eval_interval', 500)
        eval_iters       = self.config.get('eval_iters', 100)
        log_interval     = self.config.get('log_interval', 50)
        checkpoint_interval = self.config.get('checkpoint_interval', 1000)
        grad_clip        = self.config.get('grad_clip', 1.0)

        lr_config = {
            'max_lr':         self.config['learning_rate'],
            'min_lr':         self.config.get('min_lr', 3e-5),
            'warmup_iters':   self.config.get('warmup_iters', 500),
            'lr_decay_iters': self.config.get('lr_decay_iters', 45_000),
        }

        print(f"\n{'='*60}")
        print(f"  BoxingGPT Training | {max_iters:,} steps")
        print(f"{'='*60}\n")

        self.model.train()
        train_iter = iter(self.train_loader)
        t0 = time.time()
        running_loss = 0.0

        while self.step < max_iters:

            # ── LR schedule ───────────────────────────
            lr = get_lr(self.step, **lr_config)
            apply_lr(self.optimizer, lr)

            # ── Gradient Accumulation ─────────────────
            # Accumulate gradients over multiple micro-batches
            # before doing one optimizer step.
            self.optimizer.zero_grad()
            accum_loss = 0.0

            for micro_step in range(self.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)

                # Forward pass with optional mixed precision
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    _, loss = self.model(x, targets=y)

                # Scale loss by accumulation steps
                # (so the gradient magnitude is correct)
                loss = loss / self.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()

            # ── Gradient Clipping ─────────────────────
            # Clip gradient norm to prevent exploding gradients
            # Especially important in early training
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # ── Optimizer step ────────────────────────
            self.optimizer.step()
            self.step += 1
            running_loss += accum_loss

            # ── Logging ───────────────────────────────
            if self.step % log_interval == 0:
                t1 = time.time()
                avg_loss = running_loss / log_interval
                perplexity = math.exp(avg_loss)
                tokens_per_sec = (
                    log_interval *
                    self.grad_accum_steps *
                    self.config['batch_size'] *
                    self.config.get('context_length', 256)
                ) / (t1 - t0)

                print(
                    f"Step {self.step:>6}/{max_iters} | "
                    f"loss: {avg_loss:.4f} | "
                    f"ppl: {perplexity:.2f} | "
                    f"lr: {lr:.2e} | "
                    f"{tokens_per_sec/1000:.1f}K tok/s"
                )

                running_loss = 0.0
                t0 = time.time()

            # ── Evaluation ────────────────────────────
            if self.step % eval_interval == 0:
                val_loss = self.evaluate(eval_iters)
                val_ppl = math.exp(val_loss)
                print(f"\n{'─'*50}")
                print(f"  [Eval] Step {self.step} | val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.2f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(tag='best')
                    print(f"  ✓ New best validation loss!")

                print(f"{'─'*50}\n")
                self.model.train()

            # ── Periodic checkpoint ───────────────────
            if self.step % checkpoint_interval == 0:
                self.save_checkpoint()

        # ── Final save ────────────────────────────────
        self.save_checkpoint(tag='final')
        print(f"\n[Trainer] Training complete! Final val loss: {self.best_val_loss:.4f}")
