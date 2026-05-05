# 🥊 Boxing-GPT

A GPT-style Language Model built **from scratch** in PyTorch, specialized as a **Boxing & MMA Coach**.

No pretrained weights. No fine-tuning. Every component — tokenizer, attention, transformer, training loop — is written by hand.

---

## Architecture

```
Input Token IDs (B, T)
        ↓
Token Embedding + Positional Embedding     ← learned lookup tables
        ↓
Transformer Block × N                      ← the core
  ├── LayerNorm (Pre-LN)
  ├── Multi-Head Causal Self-Attention      ← scaled dot-product, causal mask
  ├── Residual Connection
  ├── LayerNorm (Pre-LN)
  ├── Feed-Forward Network (GELU)
  └── Residual Connection
        ↓
Final LayerNorm
        ↓
Linear Head → Logits (vocab_size)
        ↓
Cross-Entropy Loss / Softmax + Sample
```


## Setup

```bash
# 1. Clone and install
git clone <your-repo>
cd boxing-gpt
pip install -r requirements.txt

# 2. Create directories
mkdir -p data/raw data/processed data/tokenizer checkpoints
```

---

## Step-by-Step Training

### Step 1 — Get Data

Put your boxing/MMA text files (`.txt`) into `data/raw/`.

**Recommended datasets:**

| Source | What to get | How |
|--------|------------|-----|
| Project Gutenberg | "The Art of Boxing" (1913), "Seconds Out" | `wget https://gutenberg.org/...` |
| Wikipedia | Boxing techniques, MMA articles | `wikiextractor` tool |
| Reddit | r/boxing, r/MMA posts via Pushshift | Academic Torrents |
| YouTube | Auto-captions from coaching channels | `yt-dlp --write-auto-sub` |
| Podcasts | MMA/boxing show transcripts | Manual download or `whisper` |

**Target corpus size:** 10–50 million characters (bigger = better model)

### Step 2 — Prepare Data

```bash
# Generate sample data (for testing the pipeline)
python scripts/prepare_data.py --sample

# Process real data from data/raw/
python scripts/prepare_data.py \
    --input_dir  data/raw/ \
    --output_dir data/processed/ \
    --min_chars  100
```

### Step 3 — Train Tokenizer

```bash
python scripts/train_tokenizer.py \
    --corpus     data/processed/corpus.txt \
    --save_dir   data/tokenizer/ \
    --vocab_size 8000 \
    --test
```

### Step 4 — Train the Model

```bash
# Start training (uses config.yaml)
python scripts/train.py

# On CPU (slow, for testing)
python scripts/train.py --device cpu --max_iters 100

# On GPU
python scripts/train.py --device cuda

# Resume from checkpoint
python scripts/train.py --resume checkpoints/checkpoint_step10000.pt
```

### Step 5 — Generate Text

```bash
# Single prompt
python -m src.inference.generate \
    --checkpoint checkpoints/checkpoint_step50000_best.pt \
    --tokenizer  data/tokenizer/ \
    --prompt     "How do I improve my jab?" \
    --temperature 0.8

# Interactive chat mode
python -m src.inference.generate \
    --checkpoint checkpoints/checkpoint_step50000_best.pt \
    --tokenizer  data/tokenizer/ \
    --interactive
```

---

## Key Concepts

### Tokenization (BPE)
Byte-Pair Encoding builds a vocabulary by starting with characters and iteratively merging the most frequent adjacent pairs. After training, "jab" might become one token, while rare words are broken into subwords.

### Self-Attention
For each token position, the model computes how much it should "attend to" every other position. The causal mask ensures position i can only attend to positions 0..i (no future peeking). This is how the model learns context.

### Positional Encoding
Transformers are order-invariant by default. We add learned position vectors to tell the model WHERE each token is in the sequence.

### Residual Connections
`x = x + Layer(x)` — allows gradients to flow freely during backprop. Without these, deep networks fail to train.

### Pre-Layer Normalization
Normalizing BEFORE each sub-layer (Pre-LN) is more stable than the original post-norm. Used in GPT-2 and later models.

### Gradient Accumulation
Instead of one giant batch (too much memory), accumulate gradients over N micro-batches before updating. Effective batch size = `batch_size × grad_accum_steps`.

---

## File Reference

```
boxing-gpt/
├── configs/config.yaml          # All hyperparameters — start here
├── src/
│   ├── tokenizer/bpe.py         # BPE tokenizer from scratch
│   ├── model/
│   │   ├── embedding.py         # Token + Positional embeddings
│   │   ├── attention.py         # Multi-head causal self-attention
│   │   ├── block.py             # Transformer block (attention + FFN)
│   │   └── gpt.py               # Full GPT model + generation
│   ├── training/
│   │   ├── dataset.py           # TextDataset + DataLoader
│   │   ├── scheduler.py         # Cosine LR with warmup
│   │   └── trainer.py           # Full training loop
│   └── inference/generate.py    # Text generation CLI
├── scripts/
│   ├── prepare_data.py          # Clean + merge raw text
│   ├── train_tokenizer.py       # Train BPE tokenizer
│   └── train.py                 # Main training entry point
└── requirements.txt
```

---

## Troubleshooting

**CUDA out of memory:**
- Reduce `batch_size` in config.yaml
- Reduce `context_length` (256 → 128)
- Reduce model size (nano config)

**Loss not decreasing:**
- Check learning rate (try 1e-3 or 1e-4)
- Ensure data is clean (run prepare_data.py again)
- Increase warmup_iters if loss spikes early

**Gibberish output:**
- Model needs more training (lower loss before generating)
- Reduce temperature during inference (0.5–0.7)
- Get more/better training data

---

## License

MIT — build, train, share freely. 🥊
