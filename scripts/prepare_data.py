"""
Boxing-GPT — Data Preparation Script
======================================
Run this FIRST before training tokenizer or model.

What this does:
  1. Reads all raw .txt files from data/raw/
  2. Cleans text (removes garbage, normalizes whitespace)
  3. Writes a single merged corpus to data/processed/corpus.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHERE TO GET YOUR BOXING/MMA DATASETS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. BOOKS (via Project Gutenberg or Open Library — free/legal):
   - "The Sweet Science" by A.J. Liebling (public domain)
   - "Boxing: The American Martial Art" (instructional)
   - "The Art of Boxing" by Jim Driscoll (1913, public domain)
   Search: https://www.gutenberg.org/ebooks/search/?query=boxing

2. WIKIPEDIA DUMPS:
   - Download boxing/MMA Wikipedia articles as XML dump
   - Tool: https://github.com/attardi/wikiextractor
   - Topics: List of boxing techniques, MMA striking, grappling, etc.

3. REDDIT (via Pushshift / AcademicTorrents):
   - Subreddits: r/boxing, r/MMA, r/amateur_boxing, r/martialarts
   - Contains real coaching advice, technique discussions, strategy

4. YOUTUBE TRANSCRIPTS:
   - Use yt-dlp to download auto-captions from:
     * Precision Striking (YouTube)
     * Coach Anthony (YouTube boxing technique)
     * MMA Shredded
   - yt-dlp --write-auto-sub --sub-lang en --skip-download [URL]

5. FIGHT COMMENTARY & ANALYSIS:
   - Copy transcripts from boxing podcast episodes
   - ESPN Boxing, The Athletic boxing coverage

6. MANUALS & GUIDES:
   - Amateur boxing federation training manuals (AIBA)
   - Freely available PDF guides — extract with pdfminer

Put ALL downloaded text files into: data/raw/

Usage:
    python scripts/prepare_data.py \\
        --input_dir  data/raw/ \\
        --output_dir data/processed/ \\
        --min_chars  100
"""

import os
from pydoc import html
import re
import argparse
import unicodedata
import html
from pathlib import Path


# ─────────────────────────────────────────────
#  Text cleaning utilities
# ─────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalize unicode to ASCII where possible."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def clean_text(text: str) -> str:
    """
    Clean raw text for language model training.
    Preserves natural sentence/paragraph structure.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Add after the URL removal block in clean_text():
    text = html.unescape(text)   # converts &gt; → >, &amp; → &, etc.
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Remove excessive whitespace within lines
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove lines that are just numbers or symbols (e.g., timestamps)
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        # Skip very short lines, lines that are just numbers
        if len(line) < 3:
            continue
        if re.match(r'^[\d\s\W]+$', line):
            continue
        clean_lines.append(line)

    text = '\n'.join(clean_lines)

    # Normalize multiple blank lines to a single blank line (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def filter_boxing_relevant(text: str, min_chars: int = 100) -> list[str]:
    """
    Split text into paragraphs and keep those that are:
      - Long enough (min_chars characters)
      - Optionally, contain boxing/MMA keywords (optional filter)

    Returns list of clean paragraph strings.
    """
    paragraphs = text.split('\n\n')
    kept = []

    for para in paragraphs:
        para = para.strip()
        if len(para) < min_chars:
            continue
        kept.append(para)

    return kept


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────

def prepare_corpus(
    input_dir: str,
    output_dir: str,
    min_chars: int = 100,
    max_docs: int = None,
) -> None:
    """
    Process all .txt files in input_dir and merge into a single corpus.

    Args:
        input_dir : directory containing raw .txt files
        output_dir: directory to write processed corpus
        min_chars : minimum paragraph length to keep
        max_docs  : limit number of files processed (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'corpus.txt')

    txt_files = sorted(Path(input_dir).glob('**/*.txt'))
    if max_docs:
        txt_files = txt_files[:max_docs]

    print(f"[Prepare] Found {len(txt_files)} .txt files in {input_dir}")

    total_paragraphs = 0
    total_chars = 0

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for i, path in enumerate(txt_files):
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    raw = f.read()
            except Exception as e:
                print(f"  [!] Could not read {path}: {e}")
                continue

            # Clean
            cleaned = clean_text(raw)
            paragraphs = filter_boxing_relevant(cleaned, min_chars=min_chars)

            if not paragraphs:
                continue

            # Write paragraphs separated by double newlines
            for para in paragraphs:
                out_f.write(para + '\n\n')

            total_paragraphs += len(paragraphs)
            total_chars += sum(len(p) for p in paragraphs)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(txt_files)}] {total_paragraphs:,} paragraphs | "
                      f"{total_chars/1e6:.1f}M chars")

    print(f"\n[Prepare] Done!")
    print(f"  Output: {output_path}")
    print(f"  Total paragraphs: {total_paragraphs:,}")
    print(f"  Total characters: {total_chars:,} ({total_chars/1e6:.2f}M)")
    print(f"\n  💡 Tip: You want at least 5M characters for a meaningful model.")
    print(f"         10-50M characters is ideal for a domain-specific LLM.")


def create_sample_data(output_dir: str) -> None:
    """
    Create a small sample boxing corpus for testing the pipeline.
    Replace this with real data for actual training.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_path = os.path.join(output_dir, 'sample_boxing.txt')

    sample_texts = [
        """The jab is the most important punch in boxing. It is thrown with the lead hand and serves multiple purposes: measuring distance, setting up combinations, controlling the pace of the fight, and disrupting the opponent's rhythm. A good jab keeps your opponent at bay and creates openings for power shots.

To throw a proper jab, start from your guard position. Extend your lead hand straight out toward the target while rotating your fist so the palm faces down at full extension. Simultaneously, push off your back foot and rotate your hips slightly. The jab should snap back to guard immediately after contact.

Common jab variations include the double jab, the jab to the body, the check hook jab, and the pawing jab used purely for range finding. Each serves a different strategic purpose and should be practiced until they feel natural.""",

        """The cross is your power punch, thrown with the rear hand. When you throw a cross, you are committing your hips and shoulders fully to the punch. Start from your guard. Drive your rear fist straight forward, rotating your hip and shoulder into the punch. Pivot on your rear foot so your heel rises. Your lead hand should come back to protect your chin during this movement.

The cross generates power through the kinetic chain: feet, legs, hips, torso, shoulder, arm. A weak cross usually means you are using arm strength only. Focus on hip rotation and foot pivot to maximize power transfer.

The 1-2 combination (jab-cross) is the foundation of boxing offense. Master this before learning any other combination. Throw the jab to set up the cross, and ensure your jab hand returns to guard before or as the cross lands.""",

        """Defense in boxing is not passive. Active defense means controlling range, using angles, and making your opponent miss while you are already in position to counter. The slip is fundamental: as a straight punch comes toward your head, rotate your torso to move your head offline while keeping your weight balanced. Slip to the outside of a jab and you are in perfect position for a right hand counter.

The roll or bob-and-weave is used against hooks. Bend your knees and drop your level slightly as the hook travels over your head, then return to guard. Do not duck straight down, as this leaves you vulnerable to uppercuts. Move in a U-shape: dip, move under the punch, and come up on the other side.

Footwork is your primary defensive tool. Maintain the proper distance from your opponent. Circle away from their power hand. Step off at angles rather than retreating straight back, which is the slowest and most predictable form of retreat.""",

        """In MMA, the clinch is a transitional zone between striking and grappling. Understanding clinch work separates skilled fighters from brawlers. When you tie up with an opponent, assess your options: can you land short elbows, knee strikes, or trips? Can you off-balance them against the cage?

The Muay Thai clinch, or plum position, is among the most effective controlling positions in combat sports. Secure both hands behind the opponent's neck with your elbows pinching inward on their shoulders. From here you can throw knees to the body and head, break their posture, and control their movement. Resist the clinch by keeping your chin down and your elbows inside their arms.

Wrestling fundamentals apply in the clinch. The double underhook position gives you control of the opponent's hips and allows takedowns. From double underhooks, you can execute a body lock takedown by lifting and rotating, or a trip by placing your foot behind their heel as you drive forward.""",

        """Training periodization is the systematic planning of your athletic development over time. A typical boxing training cycle leading to a fight has three phases: general preparation, specific preparation, and peak/taper.

In the general preparation phase (8 to 12 weeks out), focus on building your aerobic base and strength. Run long distances at a conversational pace. Do heavy compound lifting: squats, deadlifts, overhead press. Spar lightly with an emphasis on technique.

In the specific preparation phase (4 to 8 weeks out), shift to boxing-specific conditioning. Replace long slow runs with interval training and hill sprints. Replace heavy lifting with explosive movements: medicine ball throws, plyometrics, kettlebell swings. Increase sparring intensity and rounds. Work on your game plan for the specific opponent.

In the peak phase (1 to 3 weeks out), reduce training volume while maintaining intensity. This is the taper. Your body needs time to recover and absorb the training. Light technical work, short sharp sparring sessions, and rest. Trust the process and do not try to cram more work in. You cannot get significantly fitter in the final week, but you can definitely fatigue yourself.""",
    ]

    with open(sample_path, 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text.strip() + '\n\n')

    print(f"[Prepare] Sample data written to {sample_path}")
    print(f"          ({len(sample_texts)} documents, {sum(len(t) for t in sample_texts):,} chars)")
    print(f"          NOTE: This is for pipeline testing only. Use real data for training.")


def main():
    parser = argparse.ArgumentParser(description="Prepare Boxing-GPT training corpus")
    parser.add_argument('--input_dir',  type=str, default='data/raw/')
    parser.add_argument('--output_dir', type=str, default='data/processed/')
    parser.add_argument('--min_chars',  type=int, default=100)
    parser.add_argument('--max_docs',   type=int, default=None)
    parser.add_argument('--sample',     action='store_true',
                        help="Create sample data for testing (no real data needed)")
    args = parser.parse_args()

    if args.sample:
        create_sample_data(args.input_dir)

    prepare_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_chars=args.min_chars,
        max_docs=args.max_docs,
    )


if __name__ == '__main__':
    main()
