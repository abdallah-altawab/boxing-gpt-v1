"""
Boxing-GPT — Main Training Script
====================================
Run this THIRD (after prepare_data.py and train_tokenizer.py).

This script:
  1. Loads config from configs/config.yaml
  2. Loads (or tokenizes) the dataset
  3. Builds the GPT model
  4. Runs the training loop
  5. Saves checkpoints

Usage:
    # Basic training
    python scripts/train.py

    # Override config values
    python scripts/train.py --device cuda --batch_size 64

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/checkpoint_step10000.pt

    # Quick test run (1 minute)
    python scripts/train.py --max_iters 100 --device cpu

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HARDWARE REQUIREMENTS (rough estimates)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 nano  (384d, 6L):  ~10M params  →  CPU ok, GPU fast
 small (768d, 12L): ~85M params  →  8GB+ VRAM recommended
 
 GPU time estimates for nano on boxing corpus (~20M tokens):
   RTX 3090 / A100: 1-2 hours
   RTX 3060:        4-6 hours
   CPU only:        several days (reduce batch size)
"""

import os
import sys
import argparse
import yaml
import torch

# Project root = the directory that contains this script's parent directory.
# Works whether the script is run from the project root, scripts/, or anywhere else.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _resolve(path: str) -> str:
    """
    If `path` is relative, resolve it against the project root so the
    script works regardless of the working directory it is launched from.
    """
    if os.path.isabs(path):
        return path
    # First try as-is (respects user's CWD when they give an explicit path)
    if os.path.exists(path):
        return path
    # Fall back to project-root-relative
    candidate = os.path.join(PROJECT_ROOT, path)
    return candidate

from src.tokenizer.bpe import BPETokenizer
from src.model.gpt import BoxingGPT
from src.training.dataset import tokenize_corpus, build_dataloaders
from src.training.trainer import Trainer


# ─────────────────────────────────────────────
#  Load config
# ─────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config and flatten for convenience."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train BoxingGPT")
    parser.add_argument('--config',      type=str, default='src/configs/config.yaml',
                        help="Path to YAML config (relative to project root or absolute)")
    parser.add_argument('--device',      type=str, default=None)
    parser.add_argument('--batch_size',  type=int, default=None)
    parser.add_argument('--max_iters',   type=int, default=None)
    parser.add_argument('--resume',      type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument('--retokenize',  action='store_true',
                        help="Force re-tokenization even if .npy file exists")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────
    config_path = _resolve(args.config)
    print(f"[Main] Loading config from {config_path}")
    cfg = load_config(config_path)

    model_cfg   = cfg['model']
    train_cfg   = cfg['training']
    tok_cfg     = cfg['tokenizer']

    # Apply CLI overrides
    if args.device:
        train_cfg['device'] = args.device
    if args.batch_size:
        train_cfg['batch_size'] = args.batch_size
    if args.max_iters:
        train_cfg['max_iters'] = args.max_iters

    # Add context_length to train_cfg for logging
    train_cfg['context_length'] = model_cfg['context_length']

    # ── Device ────────────────────────────────────
    device = train_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("[Main] CUDA not available, falling back to CPU")
        device = 'cpu'
        train_cfg['device'] = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("[Main] MPS not available, falling back to CPU")
        device = 'cpu'
        train_cfg['device'] = 'cpu'

    print(f"[Main] Using device: {device}")
    if device == 'cuda':
        print(f"[Main] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Main] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load tokenizer ────────────────────────────
    tokenizer_dir = _resolve('data/tokenizer/')
    if not os.path.exists(os.path.join(tokenizer_dir, 'vocab.json')):
        print(f"\n[Main] ERROR: Tokenizer not found at {tokenizer_dir}")
        print("  Run: python scripts/train_tokenizer.py")
        sys.exit(1)

    tokenizer = BPETokenizer.load(tokenizer_dir)
    vocab_size = len(tokenizer)
    print(f"[Main] Tokenizer loaded | vocab_size={vocab_size}")

    # Update model vocab_size to match actual tokenizer
    model_cfg['vocab_size'] = vocab_size

    # ── Prepare tokenized data ────────────────────
    corpus_path = _resolve(train_cfg['data_path'])
    token_file  = corpus_path.replace('.txt', '_tokens.npy')

    if not os.path.exists(token_file) or args.retokenize:
        print(f"\n[Main] Tokenizing corpus → {token_file}")
        tokenize_corpus(corpus_path, tokenizer, token_file)
    else:
        print(f"[Main] Found tokenized data at {token_file}")

    # ── Build DataLoaders ─────────────────────────
    train_loader, val_loader = build_dataloaders(
        token_file=token_file,
        context_length=model_cfg['context_length'],
        batch_size=train_cfg['batch_size'],
        train_split=train_cfg.get('train_split', 0.95),
    )

    # ── Build Model ───────────────────────────────
    print(f"\n[Main] Building BoxingGPT ...")
    print(f"  vocab_size={vocab_size}, context_length={model_cfg['context_length']}")
    print(f"  n_layers={model_cfg['n_layers']}, n_heads={model_cfg['n_heads']}")
    print(f"  d_model={model_cfg['d_model']}, d_ff={model_cfg['d_ff']}")

    model = BoxingGPT(
        vocab_size=vocab_size,
        context_length=model_cfg['context_length'],
        n_layers=model_cfg['n_layers'],
        n_heads=model_cfg['n_heads'],
        d_model=model_cfg['d_model'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg.get('dropout', 0.1),
    )

    # ── Build Trainer & Run ───────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
    )

    trainer.train(
        max_iters=train_cfg.get('max_iters'),
        resume_from=args.resume,
    )

    print("\n🥊 Training complete! Your BoxingGPT is ready.")
    print("Generate text with:")
    print("  python -m src.inference.generate \\")
    print("    --checkpoint checkpoints/checkpoint_best.pt \\")
    print("    --tokenizer  data/tokenizer/ \\")
    print("    --prompt     'How do I set up my right hand?' \\")
    print("    --interactive")


if __name__ == '__main__':
    main()