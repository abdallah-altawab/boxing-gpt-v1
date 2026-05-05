"""
Boxing-GPT — Evaluation & Analysis
=====================================
Tools to measure and analyze your trained model.

What's included:
  1. Perplexity evaluation on a held-out test set
  2. Sample generation at different temperatures
  3. Attention weight visualization (export to text)
  4. Token-level prediction confidence analysis
  5. Checkpoint comparison (which step was best?)

Usage:
    # Evaluate a checkpoint
    python scripts/evaluate.py \\
        --checkpoint checkpoints/checkpoint_step50000_best.pt \\
        --tokenizer  data/tokenizer/ \\
        --corpus     data/processed/corpus.txt

    # Compare multiple checkpoints
    python scripts/evaluate.py --compare \\
        --checkpoints checkpoints/checkpoint_step10000.pt \\
                      checkpoints/checkpoint_step30000.pt \\
                      checkpoints/checkpoint_step50000.pt

    # Generate samples at different temperatures
    python scripts/evaluate.py --samples --checkpoint <path>
"""

import os
import sys
import math
import json
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.gpt import BoxingGPT
from src.tokenizer.bpe import BPETokenizer
from src.training.dataset import TextDataset


# ─────────────────────────────────────────────
#  Perplexity Evaluation
# ─────────────────────────────────────────────

def compute_perplexity(
    model: BoxingGPT,
    token_file: str,
    context_length: int,
    batch_size: int = 16,
    n_batches: int = 200,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Compute perplexity on a token file (test or validation set).

    Perplexity = exp(average_cross_entropy_loss)
    Lower is better. Untrained model starts around exp(log(vocab_size)).
    A well-trained domain-specific model: 20–80 PPL.

    Args:
        model          : trained BoxingGPT
        token_file     : path to .npy token array
        context_length : model's context window size
        batch_size     : eval batch size
        n_batches      : number of batches to evaluate (set None for full eval)
        device         : compute device

    Returns:
        dict with 'loss', 'perplexity', 'bits_per_char'
    """
    from torch.utils.data import DataLoader
    model.eval()

    dataset = TextDataset(token_file, context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    total_loss = 0.0
    total_tokens = 0
    count = 0

    print(f"[Eval] Computing perplexity on {len(dataset):,} samples ...")

    with torch.no_grad():
        for x, y in loader:
            if n_batches and count >= n_batches:
                break

            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)

            # We want sum of losses, not mean, to average over all tokens
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            count += 1

            if count % 20 == 0:
                running_ppl = math.exp(total_loss / total_tokens)
                print(f"  Batch {count}/{n_batches or '?'} | Running PPL: {running_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Bits per character: useful for comparing across different vocab sizes
    # bits_per_char ≈ avg_loss / log(2) / avg_chars_per_token
    # We approximate avg chars/token ≈ 4 for English BPE
    bits_per_char = avg_loss / math.log(2) / 4.0

    results = {
        'loss': avg_loss,
        'perplexity': perplexity,
        'bits_per_char': bits_per_char,
        'n_tokens_evaluated': total_tokens,
    }

    print(f"\n[Eval] Results:")
    print(f"  Loss:          {avg_loss:.4f}")
    print(f"  Perplexity:    {perplexity:.2f}")
    print(f"  Bits/char:     {bits_per_char:.3f}")
    print(f"  Tokens eval'd: {total_tokens:,}")

    return results


# ─────────────────────────────────────────────
#  Sample Generation at Multiple Temperatures
# ─────────────────────────────────────────────

EVAL_PROMPTS = [
    "The most important punch in boxing is",
    "To improve your jab, you should",
    "When fighting a southpaw opponent,",
    "The key to developing knockout power is",
    "A good defensive fighter always",
    "Before a fight, a boxer should",
    "The liver shot is effective because",
    "To cut weight safely for a fight,",
    "Footwork is important because",
    "When an opponent clinches you,",
    "The best combination to throw after a jab is",
    "To improve your head movement,",
    "A pressure fighter uses",
    "In MMA, the clinch allows you to",
    "When your opponent drops their right hand,",
]


def generate_samples(
    model: BoxingGPT,
    tokenizer: BPETokenizer,
    device: str = 'cpu',
    temperatures: List[float] = [0.5, 0.8, 1.0, 1.2],
    max_new_tokens: int = 100,
    prompts: Optional[List[str]] = None,
    n_prompts: int = 5,
) -> Dict[str, List[Dict]]:
    """
    Generate text at multiple temperatures for quality analysis.

    Lower temperature → more deterministic, conservative
    Higher temperature → more creative, potentially incoherent

    Returns:
        Dict mapping temperature → list of {prompt, generated} dicts
    """
    model.eval()
    prompts = prompts or EVAL_PROMPTS[:n_prompts]
    results = {}

    print(f"\n[Samples] Generating at temperatures: {temperatures}")
    print(f"[Samples] {len(prompts)} prompts × {len(temperatures)} temps = "
          f"{len(prompts) * len(temperatures)} samples\n")

    for temp in temperatures:
        print(f"\n{'─'*60}")
        print(f"  Temperature: {temp}")
        print(f"{'─'*60}")
        results[str(temp)] = []

        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, add_bos=True)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=40,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_id,
                )

            full_text = tokenizer.decode(output_ids[0].tolist())

            print(f"\n  Prompt: \"{prompt}\"")
            print(f"  → {full_text[:200]}{'...' if len(full_text) > 200 else ''}")

            results[str(temp)].append({
                'prompt': prompt,
                'generated': full_text,
                'temperature': temp,
            })

    return results


# ─────────────────────────────────────────────
#  Per-Token Confidence Analysis
# ─────────────────────────────────────────────

def analyze_token_confidence(
    model: BoxingGPT,
    tokenizer: BPETokenizer,
    text: str,
    device: str = 'cpu',
) -> List[Dict]:
    """
    For each token in the text, compute:
      - The model's predicted probability of the actual next token
      - The rank of the actual token in the model's prediction
      - Top-3 predicted tokens

    This shows WHERE the model is confident vs uncertain.

    Args:
        model     : trained model
        tokenizer : tokenizer
        text      : text to analyze
        device    : compute device

    Returns:
        List of dicts, one per token position
    """
    import torch.nn.functional as F

    model.eval()
    token_ids = tokenizer.encode(text, add_bos=True)

    if len(token_ids) < 2:
        print("[Confidence] Text too short.")
        return []

    input_tensor = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)

    # Pass dummy targets to get logits for all positions (not just last)
    dummy_targets = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(input_tensor, targets=dummy_targets)  # (1, T, vocab_size)

    probs = F.softmax(logits[0], dim=-1)  # (T, vocab_size)

    results = []
    for t in range(len(token_ids) - 1):
        target_id = token_ids[t + 1]
        target_prob = probs[t, target_id].item()

        # Rank of actual token (1 = model's top prediction)
        sorted_ids = probs[t].argsort(descending=True).tolist()
        rank = sorted_ids.index(target_id) + 1

        # Top 3 predicted tokens
        top3 = [
            (tokenizer.id_to_token.get(idx, '?'), probs[t, idx].item())
            for idx in sorted_ids[:3]
        ]

        actual_token = tokenizer.id_to_token.get(target_id, '?')

        results.append({
            'position': t,
            'actual_token': actual_token,
            'actual_prob': target_prob,
            'actual_rank': rank,
            'top3': top3,
        })

    # Print summary
    print(f"\n[Confidence] Token-level analysis for:")
    print(f"  '{text[:80]}{'...' if len(text) > 80 else ''}'")
    print(f"\n  {'Token':<20} {'Prob':>8} {'Rank':>6}  Top Prediction")
    print(f"  {'─'*20} {'─'*8} {'─'*6}  {'─'*20}")

    for r in results[:20]:  # Show first 20 tokens
        top_pred = r['top3'][0][0] if r['top3'] else '?'
        mark = '✓' if r['actual_rank'] == 1 else ' '
        print(
            f"  {r['actual_token']:<20} "
            f"{r['actual_prob']:>8.4f} "
            f"{r['actual_rank']:>6}  "
            f"{top_pred} {mark}"
        )

    avg_prob = sum(r['actual_prob'] for r in results) / len(results)
    avg_rank = sum(r['actual_rank'] for r in results) / len(results)
    pct_top1 = sum(1 for r in results if r['actual_rank'] == 1) / len(results) * 100

    print(f"\n  Summary:")
    print(f"  Avg probability of actual token: {avg_prob:.4f}")
    print(f"  Avg rank of actual token:        {avg_rank:.1f}")
    print(f"  % time model's #1 prediction correct: {pct_top1:.1f}%")

    return results


# ─────────────────────────────────────────────
#  Checkpoint Comparison
# ─────────────────────────────────────────────

def compare_checkpoints(
    checkpoint_paths: List[str],
    tokenizer_dir: str,
    token_file: str,
    device: str = 'cpu',
    n_batches: int = 50,
) -> None:
    """
    Evaluate and compare multiple checkpoints to find the best one.

    Args:
        checkpoint_paths : list of .pt checkpoint files
        tokenizer_dir    : path to tokenizer
        token_file       : path to token .npy file
        device           : compute device
        n_batches        : evaluation batches per checkpoint
    """
    tokenizer = BPETokenizer.load(tokenizer_dir)
    results = []

    print(f"\n[Compare] Evaluating {len(checkpoint_paths)} checkpoints ...")

    for ckpt_path in checkpoint_paths:
        print(f"\n  Loading: {ckpt_path}")
        # Load checkpoint to get step number
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        step = ckpt.get('step', '?')
        
        model, cfg = load_model_from_checkpoint(ckpt_path, device)

        # Get context length from model
        context_length = model.embeddings.position_embedding.weight.shape[0]

        metrics = compute_perplexity(
            model, token_file,
            context_length=context_length,
            n_batches=n_batches,
            device=device,
        )
        results.append({
            'checkpoint': Path(ckpt_path).name,
            'step': step,
            **metrics
        })

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  CHECKPOINT COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Checkpoint':<40} {'Step':>8} {'PPL':>8} {'Loss':>8}")
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8}")

    best_ppl = min(r['perplexity'] for r in results)
    for r in sorted(results, key=lambda x: x['perplexity']):
        marker = ' ← BEST' if r['perplexity'] == best_ppl else ''
        print(
            f"  {r['checkpoint']:<40} "
            f"{r['step']:>8} "
            f"{r['perplexity']:>8.2f} "
            f"{r['loss']:>8.4f}"
            f"{marker}"
        )


# ─────────────────────────────────────────────
#  Model Loading Helper
# ─────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Tuple[BoxingGPT, Dict]:
    """
    Load a BoxingGPT model from a checkpoint file.

    Args:
        checkpoint_path : path to .pt checkpoint
        device          : compute device

    Returns:
        Tuple of (model, config_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    model_state = checkpoint['model_state_dict']

    # Extract model architecture from state dict
    vocab_size = model_state['embeddings.token_embedding.weight'].shape[0]
    position_emb = model_state['embeddings.position_embedding.weight']
    context_length = position_emb.shape[0]
    d_model = position_emb.shape[1]

    # Count layers from block keys like 'blocks.0.ln1.weight'
    n_layers = max(
        int(k.split('.')[1])
        for k in model_state.keys()
        if k.startswith('blocks.') and '.ln1.weight' in k
    ) + 1

    # Infer n_heads and d_ff from known patterns or defaults
    # d_model=384 with 6 heads → n_heads=6
    n_heads = 6
    # Standard ratio: d_ff = 4 * d_model
    d_ff = d_model * 4

    print(f"[Model] Loaded from step {checkpoint.get('step', '?')}")
    print(f"[Model] vocab_size={vocab_size}, context_length={context_length}, "
          f"n_layers={n_layers}, n_heads={n_heads}, d_model={d_model}")

    model = BoxingGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
    )
    model.load_state_dict(model_state)
    model.to(device)

    return model, cfg


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BoxingGPT Evaluation")
    parser.add_argument('--checkpoint',  type=str, default=None)
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None)
    parser.add_argument('--tokenizer',   type=str, default='data/tokenizer/')
    parser.add_argument('--corpus',      type=str, default='data/processed/corpus.txt')
    parser.add_argument('--device',      type=str, default='cpu')
    parser.add_argument('--n_batches',   type=int, default=200)

    # Actions
    parser.add_argument('--perplexity',  action='store_true', help="Compute perplexity")
    parser.add_argument('--samples',     action='store_true', help="Generate samples")
    parser.add_argument('--confidence',  action='store_true', help="Token confidence analysis")
    parser.add_argument('--compare',     action='store_true', help="Compare multiple checkpoints")
    parser.add_argument('--all',         action='store_true', help="Run all evaluations")

    args = parser.parse_args()

    # ── Load tokenizer + model ────────────────────
    if args.compare and args.checkpoints:
        token_file = args.corpus.replace('.txt', '_tokens.npy')
        compare_checkpoints(args.checkpoints, args.tokenizer, token_file, args.device, args.n_batches)
        return

    if not args.checkpoint:
        parser.print_help()
        print("\nERROR: --checkpoint required")
        return

    # Load tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)

    # Load model from checkpoint
    model, cfg = load_model_from_checkpoint(args.checkpoint, args.device)

    # Determine context_length from model
    context_length = model.embeddings.position_embedding.weight.shape[0]

    run_all = args.all or not any([args.perplexity, args.samples, args.confidence])

    if run_all or args.perplexity:
        token_file = args.corpus.replace('.txt', '_tokens.npy')
        if os.path.exists(token_file):
            compute_perplexity(model, token_file, context_length,
                               n_batches=args.n_batches, device=args.device)
        else:
            print(f"[Eval] Token file not found: {token_file}")
            print("  Run: python scripts/train.py --max_iters 0  (to tokenize only)")

    if run_all or args.samples:
        generate_samples(model, tokenizer, args.device,
                         temperatures=[0.5, 0.8, 1.0], max_new_tokens=150)

    if run_all or args.confidence:
        test_text = "The jab is the most important punch because it controls distance and sets up the cross."
        analyze_token_confidence(model, tokenizer, test_text, args.device)


if __name__ == '__main__':
    main()
