"""
Boxing-GPT — Train BPE Tokenizer
===================================
Run this SECOND (after prepare_data.py, before train.py).

Trains a Byte-Pair Encoding tokenizer on your processed corpus
and saves the vocabulary and merge rules.

Usage:
    python scripts/train_tokenizer.py \\
        --corpus    data/processed/corpus.txt \\
        --save_dir  data/tokenizer/ \\
        --vocab_size 8000

After training, test it:
    python scripts/train_tokenizer.py --test
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer.bpe import BPETokenizer


def train_tokenizer(
    corpus_path: str,
    save_dir: str,
    vocab_size: int = 8000,
) -> BPETokenizer:
    """
    Train and save a BPE tokenizer.

    Args:
        corpus_path : path to text corpus
        save_dir    : directory to save tokenizer files
        vocab_size  : target vocabulary size

    Returns:
        Trained BPETokenizer instance
    """
    print(f"\n{'='*50}")
    print(f"  Training BPE Tokenizer")
    print(f"  Corpus: {corpus_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"{'='*50}\n")

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus_path, vocab_size=vocab_size)
    tokenizer.save(save_dir)

    return tokenizer


def test_tokenizer(tokenizer: BPETokenizer) -> None:
    """Test the tokenizer with boxing-domain text samples."""
    print("\n" + "="*50)
    print("  Tokenizer Test")
    print("="*50)

    test_sentences = [
        "The jab is the most important punch in boxing.",
        "Rotate your hips into the cross for maximum power.",
        "Slip the jab and counter with a right hand to the body.",
        "Double leg takedown, then ground and pound.",
        "Cut to the outside angle and throw the uppercut.",
    ]

    for sent in test_sentences:
        ids = tokenizer.encode(sent)
        decoded = tokenizer.decode(ids)
        compression = len(sent.split()) / len(ids)

        print(f"\nOriginal : {sent}")
        print(f"Token IDs: {ids[:15]}{'...' if len(ids) > 15 else ''}")
        print(f"Decoded  : {decoded}")
        print(f"Words/tokens ratio: {compression:.2f}x compression")


def main():
    parser = argparse.ArgumentParser(description="Train BoxingGPT BPE Tokenizer")
    parser.add_argument('--corpus',     type=str, default='data/processed/corpus.txt')
    parser.add_argument('--save_dir',   type=str, default='data/tokenizer/')
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--test',       action='store_true', help="Test existing tokenizer")
    parser.add_argument('--test_only',  action='store_true', help="Only run test, no training")
    args = parser.parse_args()

    if args.test_only:
        print(f"Loading tokenizer from {args.save_dir} ...")
        tokenizer = BPETokenizer.load(args.save_dir)
        test_tokenizer(tokenizer)
        return

    # Train
    tokenizer = train_tokenizer(
        corpus_path=args.corpus,
        save_dir=args.save_dir,
        vocab_size=args.vocab_size,
    )

    # Optionally test right after training
    if args.test:
        test_tokenizer(tokenizer)

    print(f"\n✓ Tokenizer ready!")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Saved to:   {args.save_dir}")
    print(f"\nNext step: python scripts/train.py")


if __name__ == '__main__':
    main()
