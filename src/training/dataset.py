"""
Boxing-GPT — Dataset & DataLoader
===================================
How language model training data works:

We have a HUGE flat sequence of token ids (the entire corpus tokenized).
We sample random windows of length (context_length + 1) from it.

    tokens = [t0, t1, t2, ..., t_N]   (all tokens concatenated)

    For a window starting at position i:
        input  = tokens[i   : i + context_length]     (T tokens)
        target = tokens[i+1 : i + context_length + 1] (T tokens, shifted right by 1)

    The model sees input[t] and must predict target[t] = input[t+1].
    This is "next-token prediction" — the language model objective.

We load the entire tokenized corpus into memory as a flat tensor.
This is fine for datasets up to ~1GB of text (hundreds of millions of tokens).
For larger datasets, you'd use memory-mapped files (mmap).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional

from ..tokenizer.bpe import BPETokenizer


# ─────────────────────────────────────────────
#  Tokenize & save corpus to .bin file
# ─────────────────────────────────────────────

def tokenize_corpus(
    corpus_path: str,
    tokenizer: BPETokenizer,
    output_path: str,
    chunk_size: int = 100_000
) -> None:
    """
    Tokenize an entire text corpus and save as a flat binary array of uint16.

    uint16 supports vocab up to 65,535 — fine for our BPE vocab.
    Saving as binary is much faster to load than re-tokenizing each run.

    Args:
        corpus_path : path to raw text file
        tokenizer   : trained BPETokenizer
        output_path : where to save the .bin file
        chunk_size  : number of characters to process per chunk
    """
    print(f"[Dataset] Tokenizing {corpus_path} ...")

    all_ids = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Add BOS at start of each document/paragraph
    # We split on double newlines to get individual documents
    documents = text.split('\n\n')
    total = len(documents)

    for i, doc in enumerate(documents):
        doc = doc.strip()
        if not doc:
            continue
        ids = tokenizer.encode(doc, add_bos=True, add_eos=True)
        all_ids.extend(ids)

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{total}] {len(all_ids):,} tokens so far ...")

    # Save as numpy uint16 binary
    arr = np.array(all_ids, dtype=np.uint16)
    np.save(output_path, arr)
    print(f"[Dataset] Saved {len(arr):,} tokens → {output_path}")


# ─────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Dataset that samples (input, target) windows from a flat token array.

    Args:
        token_file     : path to .npy file with token ids
        context_length : number of tokens per sample
    """

    def __init__(self, token_file: str, context_length: int):
        super().__init__()
        self.context_length = context_length

        # Load flat token array into memory
        print(f"[Dataset] Loading tokens from {token_file} ...")
        data = np.load(token_file)
        # Convert to torch tensor (int64 for embedding compatibility)
        self.data = torch.from_numpy(data.astype(np.int64))

        # Number of possible starting positions
        self.n_samples = len(self.data) - context_length - 1
        print(f"[Dataset] Loaded {len(self.data):,} tokens | {self.n_samples:,} samples")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x : token ids for positions [idx, idx + context_length)   — LongTensor (T,)
            y : token ids for positions [idx+1, idx + context_length+1) — LongTensor (T,)
        """
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


# ─────────────────────────────────────────────
#  Build DataLoaders
# ─────────────────────────────────────────────

def build_dataloaders(
    token_file: str,
    context_length: int,
    batch_size: int,
    train_split: float = 0.95,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Args:
        token_file     : path to .npy token file
        context_length : sequence length
        batch_size     : batch size
        train_split    : fraction of data for training
        num_workers    : DataLoader worker processes

    Returns:
        (train_loader, val_loader)
    """
    dataset = TextDataset(token_file, context_length)

    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"[Dataset] Train: {len(train_dataset):,} samples | Val: {len(val_dataset):,} samples")
    return train_loader, val_loader
