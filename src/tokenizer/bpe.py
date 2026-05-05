"""
Boxing-GPT — Byte-Pair Encoding (BPE) Tokenizer
================================================
Built from scratch. No HuggingFace, no sentencepiece.

How BPE works:
  1. Start with character-level vocabulary
  2. Count all adjacent symbol pairs in corpus
  3. Merge the most frequent pair into a new symbol
  4. Repeat for `num_merges` iterations
  5. Vocabulary = initial chars + all merge rules

Usage:
    tokenizer = BPETokenizer()
    tokenizer.train("data/processed/corpus.txt", vocab_size=8000)
    tokenizer.save("data/tokenizer/")

    tokenizer = BPETokenizer.load("data/tokenizer/")
    ids = tokenizer.encode("jab cross hook uppercut")
    text = tokenizer.decode(ids)
"""

import os
import re
import json
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────
#  Helper: pre-tokenize text into "words"
#  We split on whitespace and keep punctuation
#  Each word is represented as a tuple of chars
#  with a special end-of-word marker </w>
# ─────────────────────────────────────────────

def _pre_tokenize(text: str) -> List[str]:
    """Split text into whitespace-delimited tokens."""
    return re.findall(r'\S+', text)


def _word_to_symbols(word: str) -> Tuple[str, ...]:
    """Convert a word string to a tuple of char symbols + </w> marker."""
    return tuple(list(word) + ['</w>'])


def _get_vocab(corpus_words: List[str]) -> Dict[Tuple, int]:
    """
    Count frequency of each unique word in the corpus,
    represented as a tuple of symbols.
    """
    vocab: Dict[Tuple, int] = defaultdict(int)
    for word in corpus_words:
        symbols = _word_to_symbols(word)
        vocab[symbols] += 1
    return dict(vocab)


def _get_stats(vocab: Dict[Tuple, int]) -> Dict[Tuple, int]:
    """Count frequency of every adjacent symbol pair across all words."""
    pairs: Dict[Tuple, int] = defaultdict(int)
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


def _merge_vocab(
    best_pair: Tuple[str, str],
    vocab: Dict[Tuple, int]
) -> Dict[Tuple, int]:
    """
    Merge all occurrences of `best_pair` in the vocabulary.
    E.g. ('j', 'a') → 'ja'  for every word that contains j a adjacent.
    """
    new_vocab: Dict[Tuple, int] = {}
    bigram = best_pair
    merged = ''.join(bigram)

    for symbols, freq in vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == bigram:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq

    return new_vocab


# ─────────────────────────────────────────────
#  BPETokenizer class
# ─────────────────────────────────────────────

class BPETokenizer:
    """
    A from-scratch Byte-Pair Encoding tokenizer.

    Attributes:
        vocab_size   : target vocabulary size
        merges       : ordered list of (a, b) merge rules
        token_to_id  : dict mapping token string → integer id
        id_to_token  : dict mapping integer id → token string
        special_tokens: dict of special token names → strings
    """

    # Default special tokens
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Will be set after training
        self._bpe_cache: Dict[str, Tuple[str, ...]] = {}
        self._merge_ranks: Dict[Tuple[str, str], int] = {}

    # ── Training ──────────────────────────────

    def train(self, corpus_path: str, vocab_size: Optional[int] = None) -> None:
        """
        Train BPE on a text file.

        Args:
            corpus_path: path to plain text file
            vocab_size : override self.vocab_size if provided
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        print(f"[BPE] Reading corpus from {corpus_path} ...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Step 1: collect all unique characters as base vocabulary
        words = _pre_tokenize(text)
        print(f"[BPE] Corpus: {len(text):,} chars | {len(words):,} word tokens")

        # Base vocab: all unique chars + special symbols
        base_chars: set = set()
        for word in words:
            base_chars.update(list(word))
        base_chars.add('</w>')

        # Assign ids: specials first, then chars, then merges will grow vocab
        special = [self.PAD, self.UNK, self.BOS, self.EOS]
        all_tokens = special + sorted(base_chars)

        self.token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        # Step 2: build frequency table
        vocab = _get_vocab(words)

        # Step 3: iteratively merge
        num_merges = self.vocab_size - len(self.token_to_id)
        print(f"[BPE] Starting vocab size: {len(self.token_to_id)} | Target merges: {num_merges}")

        for i in range(num_merges):
            stats = _get_stats(vocab)
            if not stats:
                print("[BPE] No more pairs to merge. Stopping early.")
                break

            best = max(stats, key=stats.get)
            vocab = _merge_vocab(best, vocab)
            self.merges.append(best)

            # Add merged token to vocabulary
            merged_token = ''.join(best)
            if merged_token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[merged_token] = new_id
                self.id_to_token[new_id] = merged_token

            if (i + 1) % 500 == 0 or i == num_merges - 1:
                print(f"[BPE] Merge {i+1}/{num_merges} | Vocab size: {len(self.token_to_id)}")

        # Build merge rank lookup for fast encoding
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._bpe_cache = {}

        print(f"[BPE] Training complete. Final vocab size: {len(self.token_to_id)}")

    # ── BPE Encoding of a single word ─────────

    def _bpe(self, word: str) -> Tuple[str, ...]:
        """Apply BPE merge rules to a single word."""
        if word in self._bpe_cache:
            return self._bpe_cache[word]

        symbols = _word_to_symbols(word)

        while True:
            # Find the best (lowest rank) adjacent pair
            pairs = {
                (symbols[i], symbols[i + 1])
                for i in range(len(symbols) - 1)
            }
            eligible = {p: self._merge_ranks[p] for p in pairs if p in self._merge_ranks}

            if not eligible:
                break

            best = min(eligible, key=eligible.get)
            merged = ''.join(best)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = tuple(new_symbols)

        self._bpe_cache[word] = symbols
        return symbols

    # ── Public API ────────────────────────────

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode a text string into a list of integer token IDs.

        Args:
            text    : input string
            add_bos : prepend <BOS> token
            add_eos : append <EOS> token

        Returns:
            List of integer token IDs
        """
        ids: List[int] = []

        if add_bos:
            ids.append(self.token_to_id[self.BOS])

        words = _pre_tokenize(text)
        for word in words:
            subwords = self._bpe(word)
            for sw in subwords:
                if sw in self.token_to_id:
                    ids.append(self.token_to_id[sw])
                else:
                    ids.append(self.token_to_id[self.UNK])

        if add_eos:
            ids.append(self.token_to_id[self.EOS])

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of integer token IDs back to a string.

        Args:
            ids: list of token IDs

        Returns:
            Decoded string
        """
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, self.UNK)
            # Skip special tokens in output
            if tok in (self.PAD, self.BOS, self.EOS):
                continue
            tokens.append(tok)

        # Reconstruct: </w> marks end-of-word (add space after)
        text = ''
        for tok in tokens:
            if tok.endswith('</w>'):
                text += tok[:-4] + ' '
            else:
                text += tok

        return text.strip()

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    def __len__(self) -> int:
        return len(self.token_to_id)

    # ── Save / Load ───────────────────────────

    def save(self, directory: str) -> None:
        """Save tokenizer files to directory."""
        os.makedirs(directory, exist_ok=True)

        # Save vocab
        vocab_path = os.path.join(directory, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

        # Save merges
        merges_path = os.path.join(directory, 'merges.json')
        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(self.merges, f, ensure_ascii=False)

        print(f"[BPE] Saved tokenizer to {directory}")

    @classmethod
    def load(cls, directory: str) -> 'BPETokenizer':
        """Load a saved tokenizer from directory."""
        tokenizer = cls.__new__(cls)
        tokenizer._bpe_cache = {}

        # Load vocab
        vocab_path = os.path.join(directory, 'vocab.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokenizer.token_to_id = json.load(f)

        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        tokenizer.vocab_size = len(tokenizer.token_to_id)

        # Load merges
        merges_path = os.path.join(directory, 'merges.json')
        with open(merges_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        tokenizer.merges = [tuple(pair) for pair in raw]
        tokenizer._merge_ranks = {
            tuple(pair): rank for rank, pair in enumerate(tokenizer.merges)
        }

        print(f"[BPE] Loaded tokenizer | Vocab size: {len(tokenizer.token_to_id)}")
        return tokenizer
