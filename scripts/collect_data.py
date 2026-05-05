"""
Boxing-GPT — Data Collection Script (v3)
==========================================
Collects boxing text from free, legal, no-API-key sources.
Boxing-focused (not MMA). All scraping errors handled cleanly.

Sources:
  1. Wikipedia          — boxing articles via public REST API
  2. YouTube            — auto-generated transcripts via yt-dlp
  3. Reddit             — top posts + comments (boxing subreddits only)
  4. Stack Exchange     — Martial Arts Q&A via public API
  5. Generated data     — expert glossary + coaching Q&A pairs

Changes vs v2:
  - YouTube: stderr suppressed, dead channels removed, new channels added
  - Reddit:  retry logic + exponential backoff for rate-limit recovery
  - StackExchange: gzip decompression fixed (API always returns gzip)
  - Wikipedia: redirect-target deduplication (no more duplicate 51K-char articles)
  - Focus shifted to boxing (MMA/BJJ/wrestling de-prioritised)

Usage:
    python scripts/collect_data.py --all
    python scripts/collect_data.py --wikipedia
    python scripts/collect_data.py --youtube
    python scripts/collect_data.py --reddit
    python scripts/collect_data.py --stackexchange
    python scripts/collect_data.py --glossary

Each collector writes .txt files to data/raw/<source>/
Next: python scripts/prepare_data.py
"""

import gzip
import os
import re
import sys
import json
import time
import random
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Set


RAW_DIR = Path("data/raw")

# Reddit requires a descriptive User-Agent or it blocks requests
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (compatible; BoxingGPT-DataBot/3.0; '
        '+https://github.com/your-repo/boxing-gpt; '
        'educational-research)'
    )
}


# ──────────────────────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────────────────────

def fetch_url(url: str, timeout: int = 20, retries: int = 3) -> str:
    """
    Fetch a URL and return the response body as a string.

    Handles:
      - gzip / deflate encoding (Stack Exchange always returns gzip)
      - automatic retry with exponential backoff on transient failures
    """
    delay = 2.0
    last_exc: Exception = RuntimeError("unknown error")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            req.add_header('Accept-Encoding', 'gzip, deflate')
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                enc = resp.headers.get('Content-Encoding', '').lower()
                if enc == 'gzip':
                    raw = gzip.decompress(raw)
                elif enc == 'deflate':
                    import zlib
                    raw = zlib.decompress(raw)
                return raw.decode('utf-8', errors='replace')
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                jitter = random.uniform(0, delay * 0.3)
                time.sleep(delay + jitter)
                delay *= 2          # exponential backoff

    raise last_exc


def safe_filename(name: str, max_len: int = 80) -> str:
    """Convert an arbitrary string to a safe filename."""
    name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
    return name[:max_len]


# ══════════════════════════════════════════════════════════════
#  1. WIKIPEDIA COLLECTOR
#  Boxing-focused. Redirect-deduplication: we track which
#  canonical page-id we have already saved and skip duplicates.
#  (Many article titles redirect to the same Wikipedia page.)
# ══════════════════════════════════════════════════════════════

WIKIPEDIA_ARTICLES = [
    # ── Core boxing techniques ───────────────────────────────
    "Boxing styles and techniques",
    "Jab",
    "Cross (boxing)",
    "Hook (boxing)",
    "Uppercut",
    "Overhand (boxing)",
    "Bolo punch",
    "Superman punch",
    "Body shot (boxing)",
    "Combination (boxing)",
    "Counter-punching",

    # ── Stances & styles ────────────────────────────────────
    "Orthodox stance",
    "Southpaw stance",
    "Peek-a-boo style",
    "Out-boxer",
    "Swarmer",
    "Slugger",
    "Pressure fighting",

    # ── Defence ──────────────────────────────────────────────
    "Slip (boxing)",
    "Bob and weave",
    "Clinch (boxing)",
    "Blocking (martial arts)",
    "Parry (boxing)",
    "Footwork (martial arts)",
    "Shoulder roll",
    "Pull counter",
    "In-fighting",

    # ── Training equipment & methods ─────────────────────────
    "Boxing training",
    "Shadowboxing",
    "Speed bag",
    "Heavy bag",
    "Sparring",
    "Focus mitt",
    "Jump rope",
    "Roadwork (boxing)",
    "Strength and conditioning",

    # ── Ring craft & strategy ─────────────────────────────────
    "Boxing strategy",
    "Ring generalship",
    "Cut man",
    "Cornerman",
    "Southpaw boxing",

    # ── Conditioning science ──────────────────────────────────
    "Sports periodization",
    "High-intensity interval training",
    "VO2 max",
    "Lactic acid",
    "Strength training",
    "Plyometrics",
    "Weight cutting",
    "Sports nutrition",
    "Overtraining syndrome",
    "Recovery (sport)",

    # ── Legendary boxers ──────────────────────────────────────
    "Muhammad Ali",
    "Mike Tyson",
    "Floyd Mayweather Jr.",
    "Manny Pacquiao",
    "Sugar Ray Robinson",
    "Joe Louis",
    "Rocky Marciano",
    "George Foreman",
    "Joe Frazier",
    "Vasyl Lomachenko",
    "Sugar Ray Leonard",
    "Marvin Hagler",
    "Thomas Hearns",
    "Roberto Duran",
    "Oscar De La Hoya",
    "Lennox Lewis",
    "Evander Holyfield",
    "Wladimir Klitschko",
    "Vitali Klitschko",
    "Terence Crawford",
    "Errol Spence Jr.",
    "Canelo Alvarez",
    "Gennady Golovkin",
    "Andre Ward",
    "James Toney",
    "Pernell Whitaker",
    "Aaron Pryor",
    "Julio César Chávez",
    "Erik Morales",
    "Marco Antonio Barrera",
    "Bernard Hopkins",
    "Roy Jones Jr.",
    "Shane Mosley",
    "Oscar Larios",
    "Antonio Margarito",
    "Ricky Hatton",
    "Naseem Hamed",
    "Kostya Tszyu",
    "Arturo Gatti",
    "Mickey Ward",
    "Erik Morales",
    "Juan Manuel Marquez",
    "Tim Bradley",
    "Nonito Donaire",
    "Naoya Inoue",
    "Oleksandr Usyk",
    "Anthony Joshua",
    "Tyson Fury",
    "Deontay Wilder",
    "Daniel Jacobs",
    "Jermall Charlo",
    "Jermell Charlo",
    "Vergil Ortiz Jr.",
    "Ryan Garcia",
    "Gervonta Davis",

    # ── Organisations & history ───────────────────────────────
    "International Boxing Federation",
    "World Boxing Council",
    "World Boxing Association",
    "World Boxing Organization",
    "History of boxing",
    "Amateur boxing",
    "Professional boxing",
    "Olympic boxing",
    "AIBA",
    "Golden Gloves",
    "List of boxing techniques",
    "List of boxing organizations",
    "Marquess of Queensberry Rules",
    "London Prize Ring rules",
    "Bare-knuckle boxing",
]


def _fetch_wikipedia_page(title: str) -> Dict:
    """
    Fetch Wikipedia article text + page-id via the public REST API.
    Returns dict with keys: 'pageid', 'text', 'redirected_from'.
    Returns None on failure or missing page.
    """
    encoded = urllib.parse.quote(title.replace(' ', '_'))
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={encoded}&prop=extracts"
        f"&explaintext=true&format=json&redirects=1"
    )
    try:
        raw = fetch_url(url, retries=3)
        data = json.loads(raw)
        pages = data['query']['pages']
        page = next(iter(pages.values()))
        pageid = page.get('pageid', -1)
        if 'extract' not in page or page.get('missing') or pageid == -1:
            return None
        return {'pageid': pageid, 'text': page['extract']}
    except Exception:
        return None


def _clean_wikipedia(text: str) -> str:
    """Remove Wikipedia markup artifacts and boilerplate."""
    text = re.sub(r'==+[^=]+=+', '', text)      # section headers
    text = re.sub(r'\[edit\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)           # citation markers [1]
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def collect_wikipedia(output_dir: Path, delay: float = 1.0) -> int:
    """
    Fetch boxing Wikipedia articles.
    Deduplicates by Wikipedia page-id to skip redirect aliases.
    """
    output_dir = output_dir / "wikipedia"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track page-ids we have already saved (avoids redirect duplicates)
    seen_pageids: Set[int] = set()
    # Pre-load ids from files already on disk
    for f in output_dir.glob('*.txt'):
        first_line = f.read_text(encoding='utf-8', errors='replace').split('\n')[0]
        m = re.search(r'pageid=(\d+)', first_line)
        if m:
            seen_pageids.add(int(m.group(1)))

    collected = 0
    total = len(WIKIPEDIA_ARTICLES)
    print(f"\n[Wikipedia] Collecting {total} articles (deduplication enabled) ...")

    for i, title in enumerate(WIKIPEDIA_ARTICLES):
        safe = safe_filename(title)
        out_path = output_dir / f"{safe}.txt"

        if out_path.exists():
            print(f"  [{i+1:>3}/{total}] skip (exists): {title}")
            collected += 1
            continue

        try:
            result = _fetch_wikipedia_page(title)

            if result is None:
                print(f"  [{i+1:>3}/{total}] missing/empty: {title}")
                time.sleep(delay)
                continue

            pageid = result['pageid']
            text   = result['text']

            if len(text) < 500:
                print(f"  [{i+1:>3}/{total}] too short ({len(text)} chars): {title}")
                time.sleep(delay)
                continue

            if pageid in seen_pageids:
                print(f"  [{i+1:>3}/{total}] redirect-dup (pageid={pageid}): {title}")
                time.sleep(delay)
                continue

            text = _clean_wikipedia(text)
            out_path.write_text(
                f"# {title}  [pageid={pageid}]\n\n{text}\n",
                encoding='utf-8'
            )
            seen_pageids.add(pageid)
            print(f"  [{i+1:>3}/{total}] ✓ {title} ({len(text):,} chars)")
            collected += 1
            time.sleep(delay)

        except Exception as e:
            print(f"  [{i+1:>3}/{total}] ERROR: {title} — {e}")
            time.sleep(delay * 2)

    print(f"[Wikipedia] Done: {collected}/{total} saved")
    return collected


# ══════════════════════════════════════════════════════════════
#  2. YOUTUBE TRANSCRIPT COLLECTOR
#
#  Key fix: capture stderr so yt-dlp errors do NOT flood the
#  terminal.  Per-channel error counts are shown in summary.
#
#  Channel list is boxing-focused. Channels verified working as
#  of 2026; 404 channels from v2 have been removed/replaced.
# ══════════════════════════════════════════════════════════════

YOUTUBE_CHANNELS = {
    # ── Core boxing coaching & technique ─────────────────────
    "coach_anthony": {
        "url": "https://www.youtube.com/@CoachAnthonyBoxing",
        "desc": "Amateur & pro boxing coaching, technique fundamentals",
    },
    "fightcamp": {
        "url": "https://www.youtube.com/@FightCamp",
        "desc": "Boxing workouts, technique breakdowns, training guides",
    },
    "fighttips": {
        "url": "https://www.youtube.com/@Fighttips",
        "desc": "Boxing & self-defence coaching (1M+ subscribers)",
    },
    "hard2hurt": {
        "url": "https://www.youtube.com/@hard2hurt",
        "desc": "Technical boxing analysis, gear reviews, coaching",
    },
    "tony_jeffries": {
        "url": "https://www.youtube.com/@Tony_Jeffries",
        "desc": "Olympic silver medalist — boxing coaching & technique",
    },
    "expert_boxing": {
        "url": "https://www.youtube.com/@ExpertBoxing",
        "desc": "In-depth boxing strategy, stance, technique tutorials",
    },
    "boxing_science": {
        "url": "https://www.youtube.com/@BoxingScience",
        "desc": "Science-backed boxing conditioning & training methods",
    },
    "my_boxing_coach": {
        "url": "https://www.youtube.com/@MyBoxingCoach",
        "desc": "Comprehensive boxing coaching tutorials",
    },
    "skillr_boxing": {
        "url": "https://www.youtube.com/@skillrboxing",
        "desc": "Boxing technique, drills and combinations",
    },
    "mike_rashid": {
        "url": "https://www.youtube.com/@MikeRashidOfficial",
        "desc": "Boxing training and strength conditioning",
    },
    "world_class_boxing": {
        "url": "https://www.youtube.com/@WorldClassBoxingChannel",
        "desc": "World-class boxing technique and fight breakdowns",
    },
    "precision_boxing": {
        "url": "https://www.youtube.com/@PrecisionBoxing",
        "desc": "Precision boxing technique coaching",
    },
    "lawrence_kenshin": {
        "url": "https://www.youtube.com/@LawrenceKenshin",
        "desc": "Elite striking breakdowns and boxing technique analysis",
    },

    # ── Major boxing promotions (fight commentary & analysis) ─
    "matchroom_boxing": {
        "url": "https://www.youtube.com/@MatchroomBoxing",
        "desc": "Matchroom Boxing — professional fight coverage",
    },
    "sky_sports_boxing": {
        "url": "https://www.youtube.com/SkySportsBoxing",
        "desc": "Sky Sports Boxing — professional fight coverage & analysis",
    },
    "premier_boxing_champions": {
        "url": "https://www.youtube.com/PremierBoxingChampions",
        "desc": "PBC — professional boxing fights and post-fight analysis",
    },
    "top_rank_boxing": {
        "url": "https://www.youtube.com/@toprank",
        "desc": "Top Rank Boxing — professional fights, training footage",
    },
    "queensberry_promotions": {
        "url": "https://www.youtube.com/@QueensberryPromotions",
        "desc": "Queensberry Promotions — Frank Warren's boxing channel",
    },
    "dazn_boxing": {
        "url": "https://www.youtube.com/daznboxing",
        "desc": "DAZN Boxing — fights, analysis, training content",
    },
}


def _vtt_to_text(vtt_content: str) -> str:
    """
    Convert WebVTT subtitle file to clean plain text.

    Strips timestamps, sequence numbers, HTML tags (<c>, <b>, etc.)
    and deduplicates consecutive identical lines (common in auto-captions).
    """
    text_lines: List[str] = []
    prev = ''

    for line in vtt_content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('WEBVTT') or line.startswith('NOTE') or line.startswith('STYLE'):
            continue
        if '-->' in line:
            continue
        if re.match(r'^\d+$', line):          # sequence numbers
            continue
        line = re.sub(r'<[^>]+>', '', line)   # strip <c>, <b>, etc.
        line = line.strip()
        if not line:
            continue
        if line == prev:                       # deduplicate rolling captions
            continue
        text_lines.append(line)
        prev = line

    text = ' '.join(text_lines)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _count_stderr_errors(stderr_text: str) -> Dict[str, int]:
    """Parse yt-dlp stderr and count error categories."""
    counts: Dict[str, int] = {
        'age_gate': 0,
        'members_only': 0,
        'unavailable': 0,
        'not_found': 0,
        'other': 0,
    }
    for line in stderr_text.splitlines():
        if not line.strip():
            continue
        lo = line.lower()
        if 'sign in to confirm your age' in lo:
            counts['age_gate'] += 1
        elif 'members' in lo and ('join' in lo or 'available' in lo):
            counts['members_only'] += 1
        elif 'not available' in lo or 'this video is unavailable' in lo:
            counts['unavailable'] += 1
        elif 'http error 404' in lo or 'unable to download api page' in lo:
            counts['not_found'] += 1
        elif 'error' in lo:
            counts['other'] += 1
    return counts


def collect_youtube_transcripts(
    output_dir: Path,
    max_videos: int = 500,
    channels: Optional[Dict] = None,
) -> int:
    """
    Download auto-generated English transcripts from boxing channels.

    Errors from yt-dlp are captured (not printed) and summarised per
    channel so the terminal stays readable.

    Requires: pip install yt-dlp
    """
    import subprocess

    try:
        r = subprocess.run(
            ['yt-dlp', '--version'],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            raise FileNotFoundError
        print(f"[YouTube] yt-dlp {r.stdout.strip()}")
    except FileNotFoundError:
        print("[YouTube] yt-dlp not found — install with: pip install yt-dlp")
        return 0

    channels = channels or YOUTUBE_CHANNELS
    output_dir = output_dir / "youtube"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_collected = 0
    print(f"\n[YouTube] {len(channels)} channels | max {max_videos} videos each\n")

    for key, info in channels.items():
        ch_dir = output_dir / key
        ch_dir.mkdir(exist_ok=True)

        cmd = [
            'yt-dlp',
            '--write-auto-sub',
            '--sub-lang', 'en',
            '--skip-download',
            '--sub-format', 'vtt',
            '--playlist-end', str(max_videos),
            '--output', str(ch_dir / '%(title)s.%(ext)s'),
            '--ignore-errors',
            '--no-warnings',
            '--quiet',                    # suppress normal output
            '--sleep-interval', '1',
            '--max-sleep-interval', '3',
            '--extractor-retries', '3',
            info['url'],
        ]

        # ── Run yt-dlp, capture stderr so errors don't flood terminal ──
        try:
            proc = subprocess.run(
                cmd,
                timeout=600,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,   # <── KEY FIX: capture stderr
                text=True,
            )
            err_counts = _count_stderr_errors(proc.stderr or '')
        except subprocess.TimeoutExpired:
            print(f"  [{key}] ⚠ timeout — kept partial downloads")
            total_collected += _convert_vtt_files(ch_dir)
            continue
        except Exception as e:
            print(f"  [{key}] ✗ {e}")
            continue

        # ── Convert .vtt → .txt ────────────────────────────────────────
        saved = _convert_vtt_files(ch_dir)
        total_collected += saved

        # ── Print one clean line per channel ──────────────────────────
        err_parts = []
        if err_counts['not_found'] > 0:
            err_parts.append(f"channel-not-found")
        if err_counts['age_gate'] > 0:
            err_parts.append(f"{err_counts['age_gate']} age-gated")
        if err_counts['members_only'] > 0:
            err_parts.append(f"{err_counts['members_only']} members-only")
        if err_counts['unavailable'] > 0:
            err_parts.append(f"{err_counts['unavailable']} unavailable")
        if err_counts['other'] > 0:
            err_parts.append(f"{err_counts['other']} other-err")

        err_str = f"  [{', '.join(err_parts)}]" if err_parts else ""
        print(f"  [{key}]  {saved} transcripts{err_str}")

    print(f"\n[YouTube] Done: {total_collected} total transcripts")
    return total_collected


def _convert_vtt_files(ch_dir: Path) -> int:
    """Convert all .vtt files in a directory to .txt, remove .vtt."""
    count = 0
    for vtt in list(ch_dir.glob('*.vtt')):
        try:
            raw  = vtt.read_text(encoding='utf-8', errors='replace')
            text = _vtt_to_text(raw)
            if len(text) > 300:
                vtt.with_suffix('.txt').write_text(text, encoding='utf-8')
                count += 1
            vtt.unlink()
        except Exception:
            pass
    return count


# ══════════════════════════════════════════════════════════════
#  3. REDDIT COLLECTOR
#
#  Key fix: exponential backoff on failures, longer delay between
#  subreddits, and a separate delay between comment fetches.
#  Focused on boxing subreddits only.
#
#  Note: Reddit's public JSON API rate-limits ~60 req/min.
#  Increase --delay if you see repeated failures.
# ══════════════════════════════════════════════════════════════

REDDIT_SUBREDDITS = [
    "boxing",
    "amateur_boxing",
    "Boxing_Footwork",
    "ProBoxing",
    "TechnicalBoxing",
    "combatsports",   # smaller subreddit; less rate-limit pressure
]

BOXING_KEYWORDS = {
    'technique', 'how to', 'coaching', 'training', 'jab', 'cross', 'hook',
    'uppercut', 'footwork', 'defense', 'defence', 'guard', 'sparring', 'combo',
    'combination', 'stance', 'advice', 'tips', 'drills', 'form', 'help',
    'improve', 'beginner', 'punch', 'power', 'speed', 'conditioning',
    'workout', 'strength', 'boxing', 'movement', 'counter', 'slip',
    'block', 'parry', 'southpaw', 'orthodox', 'pressure', 'ring',
    'body shot', 'liver', 'clinch', 'referee', 'corner', 'ringcraft',
}


def _reddit_get(url: str, max_retries: int = 4) -> Optional[dict]:
    """
    Fetch a Reddit JSON endpoint with retry + exponential backoff.
    Reddit's public API sometimes returns 429 or empty responses.
    """
    delay = 3.0
    for attempt in range(max_retries):
        try:
            raw = fetch_url(url, timeout=25, retries=1)
            data = json.loads(raw)
            # Reddit returns {'error': 429} on rate-limit
            if isinstance(data, dict) and data.get('error'):
                raise ValueError(f"Reddit API error {data['error']}")
            return data
        except Exception as exc:
            if attempt < max_retries - 1:
                jitter = random.uniform(0, delay * 0.4)
                time.sleep(delay + jitter)
                delay = min(delay * 2, 60)
            else:
                return None
    return None


def _is_boxing_relevant(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in BOXING_KEYWORDS)


def _extract_comments(comment_list: list, depth: int = 0, max_depth: int = 3) -> List[str]:
    texts: List[str] = []
    if depth > max_depth:
        return texts
    for item in comment_list:
        if not isinstance(item, dict):
            continue
        data = item.get('data', {})
        body = data.get('body', '')
        if body and body not in ('[deleted]', '[removed]') and len(body) > 40:
            texts.append(body)
        replies = data.get('replies', '')
        if isinstance(replies, dict):
            children = replies.get('data', {}).get('children', [])
            texts.extend(_extract_comments(children, depth + 1, max_depth))
    return texts


def collect_reddit(
    output_dir: Path,
    posts_per_sub: int = 150,
    delay: float = 2.5,
) -> int:
    """
    Collect top boxing posts + comments from Reddit subreddits.

    Endpoint: https://www.reddit.com/r/{sub}/top.json?t=all&limit=100
    No credentials required; respects rate limits via backoff.
    """
    output_dir = output_dir / "reddit"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    print(f"\n[Reddit] {len(REDDIT_SUBREDDITS)} subreddits, up to {posts_per_sub} posts each")

    for subreddit in REDDIT_SUBREDDITS:
        sub_dir = output_dir / subreddit
        sub_dir.mkdir(exist_ok=True)
        saved = 0
        after: Optional[str] = None

        print(f"\n  r/{subreddit}")

        while saved < posts_per_sub:
            limit = min(100, posts_per_sub - saved)
            url = (
                f"https://www.reddit.com/r/{subreddit}/top.json"
                f"?t=all&limit={limit}&raw_json=1"
            )
            if after:
                url += f"&after={after}"

            data = _reddit_get(url)
            if data is None:
                print(f"    ✗ could not fetch r/{subreddit} listing — skipping")
                break

            children = data.get('data', {}).get('children', [])
            if not children:
                break
            after = data.get('data', {}).get('after')

            for post in children:
                pd = post.get('data', {})
                title   = pd.get('title', '')
                post_id = pd.get('id', '')
                selftext = pd.get('selftext', '')
                score   = pd.get('score', 0)

                if score < 5 or not post_id:
                    continue
                if not _is_boxing_relevant(title):
                    continue

                out_path = sub_dir / f"{post_id}.txt"
                if out_path.exists():
                    saved += 1
                    continue

                # Fetch comments with backoff
                time.sleep(delay + random.uniform(0, 0.5))
                c_url = (
                    f"https://www.reddit.com/r/{subreddit}"
                    f"/comments/{post_id}.json?limit=50&sort=top&depth=3&raw_json=1"
                )
                c_data = _reddit_get(c_url)
                comments: List[str] = []
                if c_data and isinstance(c_data, list) and len(c_data) > 1:
                    c_children = c_data[1].get('data', {}).get('children', [])
                    comments = _extract_comments(c_children)

                lines = [f"# {title}\n"]
                if selftext and selftext not in ('[deleted]', '[removed]'):
                    lines.append(selftext.strip())
                for c in comments[:15]:
                    lines.append(f"\n---\n{c.strip()}")

                doc = '\n'.join(lines)
                if len(doc) < 120:
                    continue

                out_path.write_text(doc, encoding='utf-8')
                saved += 1
                total += 1

                if saved % 25 == 0:
                    print(f"    {saved}/{posts_per_sub} saved")

            if not after:
                break
            time.sleep(delay)

        print(f"  ✓ {saved} posts from r/{subreddit}")
        time.sleep(delay * 2)    # extra pause between subreddits

    print(f"\n[Reddit] Done: {total} files saved")
    return total


# ══════════════════════════════════════════════════════════════
#  4. STACK EXCHANGE — MARTIAL ARTS Q&A
#
#  Key fix: the Stack Exchange API always returns gzip-compressed
#  responses.  The v2 fetch_url didn't decompress them, so
#  json.loads() always raised an exception → silent empty results.
#  This is now handled in the shared fetch_url() above.
# ══════════════════════════════════════════════════════════════

STACKEXCHANGE_TAGS = {
    "martialarts": [
        "boxing", "punching", "footwork", "technique", "training",
        "sparring", "amateur-boxing", "professional-boxing",
    ],
    "fitness": [
        "boxing", "combat-sports", "conditioning", "strength-training",
    ],
    "sports": [
        "boxing",
    ],
}


def _se_fetch(endpoint: str, params: dict) -> list:
    """Fetch a Stack Exchange API endpoint. Returns items list."""
    qs = urllib.parse.urlencode(params)
    url = f"https://api.stackexchange.com/2.3/{endpoint}?{qs}"
    try:
        raw  = fetch_url(url, timeout=20, retries=3)
        data = json.loads(raw)
        return data.get('items', [])
    except Exception:
        return []


def collect_stackexchange(
    output_dir: Path,
    max_per_tag: int = 100,
    delay: float = 1.5,
) -> int:
    """
    Collect boxing Q&A from Stack Exchange.
    Each file: question title + body + top-voted answers.
    """
    output_dir = output_dir / "stackexchange"
    output_dir.mkdir(parents=True, exist_ok=True)

    def html_to_text(html: str) -> str:
        html = re.sub(r'<pre[^>]*>.*?</pre>', '', html, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', html)
        for ent, ch in [
            ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
            ('&quot;', '"'), ('&#39;', "'"), ('&nbsp;', ' '), ('&hellip;', '...'),
        ]:
            text = text.replace(ent, ch)
        return re.sub(r'\s+', ' ', text).strip()

    total = 0
    sites = list(STACKEXCHANGE_TAGS.keys())
    print(f"\n[StackExchange] {len(sites)} sites")

    for site in sites:
        site_dir = output_dir / site
        site_dir.mkdir(exist_ok=True)
        tags = STACKEXCHANGE_TAGS[site]
        print(f"\n  {site}.stackexchange.com — tags: {tags}")

        for tag in tags:
            questions = _se_fetch('questions', {
                'order': 'desc', 'sort': 'votes', 'tagged': tag,
                'site': site, 'pagesize': min(100, max_per_tag),
                'filter': 'withbody',
            })
            time.sleep(delay)

            if not questions:
                print(f"    [{tag}] no results")
                continue

            tag_count = 0
            for q in questions:
                q_id    = q.get('question_id')
                title   = q.get('title', '')
                body_h  = q.get('body', '')
                score   = q.get('score', 0)

                if not q_id or score < 1:
                    continue

                out = site_dir / f"{q_id}.txt"
                if out.exists():
                    tag_count += 1
                    continue

                answers = _se_fetch(f'questions/{q_id}/answers', {
                    'order': 'desc', 'sort': 'votes',
                    'site': site, 'filter': 'withbody',
                })
                time.sleep(delay)

                body_t = html_to_text(body_h)
                lines  = [f"# {title}\n", body_t]

                for ans in answers:
                    if ans.get('score', 0) < 1:
                        continue
                    at = html_to_text(ans.get('body', ''))
                    if len(at) > 50:
                        lines.append(f"\n## Answer (score: {ans['score']})\n{at}")

                doc = '\n'.join(lines)
                if len(doc) < 150:
                    continue

                out.write_text(doc, encoding='utf-8')
                tag_count += 1
                total += 1

            print(f"    [{tag}] {tag_count} documents")
            time.sleep(delay)

    print(f"\n[StackExchange] Done: {total} Q&A documents")
    return total


# ══════════════════════════════════════════════════════════════
#  5. GENERATED DATA — Glossary + Q&A + Training Programs
# ══════════════════════════════════════════════════════════════

BOXING_GLOSSARY = {
    "Punches": {
        "Jab": (
            "A quick, straight punch thrown with the lead hand from the guard position. "
            "The jab is the most important punch in boxing. It controls distance, sets up "
            "combinations, disrupts the opponent's rhythm, and accumulates points. Proper "
            "technique: rotate the fist palm-down at full extension, simultaneously push off "
            "the rear foot and rotate the hip slightly, then snap the hand back to guard "
            "immediately after contact. Variations: double jab, body jab, check hook jab, "
            "pawing jab (range-finding)."
        ),
        "Cross": (
            "A powerful straight punch thrown with the rear hand. Power comes from the full "
            "kinetic chain: rear foot pivot, hip rotation, shoulder rotation, and arm extension. "
            "Commonly the second punch in the 1-2 combination. During the cross, the lead hand "
            "should return to the chin guard. Common errors: dropping the lead hand, failing to "
            "pivot the rear foot, using only arm strength without hip rotation."
        ),
        "Lead Hook": (
            "A punch thrown in a semicircular arc with the lead hand, targeting the head or body. "
            "Proper technique: raise the elbow to shoulder height, rotate the entire body through "
            "the punch with the rear foot as pivot. The fist can be vertical (thumb up) or "
            "horizontal. Body hooks target the liver (opponent's right side) and floating ribs. "
            "The lead hook is extremely effective following the cross in the 1-2-3 combination."
        ),
        "Rear Hook": (
            "A hook thrown with the rear hand, less common than the lead hook but effective as a "
            "counter. Generates significant power from hip rotation. Often used after slipping "
            "inside a jab or in combination with the 3-3-2 (lead hook, lead hook, cross)."
        ),
        "Uppercut": (
            "An upward punch from either hand targeting the chin or solar plexus. Most effective "
            "at close range and in the clinch. Technique: bend the knees slightly, drop the "
            "punching shoulder, drive upward through the punch with hip and shoulder rotation. "
            "The lead uppercut is used to break a tight guard; the rear uppercut generates "
            "significant power for a potential knockout."
        ),
        "Overhand": (
            "A looping punch thrown over the opponent's guard, traveling in a downward arc. "
            "Effective against shorter opponents, peek-a-boo guard users, and fighters who "
            "circle away. The overhand right is a common power shot in boxing. "
            "Risk: it takes longer to throw than a straight punch and leaves the thrower open."
        ),
        "Body Shot": (
            "Any punch targeting the midsection. Primary targets: the liver (under the right side "
            "of the opponent's ribcage), the solar plexus, and the floating ribs. Body shots "
            "accumulate damage over rounds, slow the opponent's movement, and can produce "
            "knockdowns. The liver shot, thrown as a left hook to the body, is one of the most "
            "devastating finishing techniques in boxing."
        ),
    },
    "Defence": {
        "Slip": (
            "Rotating the torso to move the head offline, avoiding a punch without moving the "
            "feet. Slipping to the outside of a jab (to the right for an orthodox fighter) "
            "places you in position for a counter right hand. Slipping inside (left) creates "
            "opportunities for hooks and uppercuts. The key is a small, efficient movement — "
            "not a large exaggerated lean."
        ),
        "Roll (Bob and Weave)": (
            "Bending at the knees to duck under a hook punch, rising on the opposite side. "
            "The movement is U-shaped: dip the knees, move laterally under the punch arc, "
            "rise on the far side. Critical error: ducking straight down, which invites uppercuts. "
            "After rolling, you are in position to deliver a hook or uppercut of your own."
        ),
        "Parry": (
            "Redirecting an incoming punch with an open hand rather than blocking it. A jab parry "
            "uses the rear hand to push the jab offline to the inside while setting up a counter. "
            "More energy-efficient than full blocks. Requires precise timing. Variations: "
            "cross-parry, pat down (pushing the jab down to create an opening for a jab or cross)."
        ),
        "High Guard Block": (
            "Raising both hands to protect the head, absorbing or deflecting punches. The Philly "
            "shell (shoulder roll) uses the lead shoulder to absorb jabs. The peek-a-boo guard "
            "keeps both fists directly in front of the face. Less efficient than slipping but "
            "more reliable when fatigued or hurt."
        ),
        "Clinch": (
            "Tying up with the opponent at close range to prevent punching and allow recovery. "
            "Legal when both fighters are in mutual embrace; the referee separates them. "
            "Skilled use of the clinch: smother the opponent's arms, apply shoulder pressure, "
            "lean weight to exhaust them. The referee will break when no punching is possible."
        ),
        "Shoulder Roll": (
            "Using the lead shoulder to deflect punches while keeping the lead hand low. "
            "Popularised by Floyd Mayweather Jr. and James Toney. The technique allows "
            "counterpunching from a longer range. Requires precise distance management and "
            "reflexes developed through extensive mitt work and sparring."
        ),
        "Pivot": (
            "Rotating on the lead foot to change angle relative to the opponent. An outside "
            "pivot moves your body off the opponent's centre line while keeping them in front "
            "of you, creating an angle for the cross or lead hook. Pivots are fundamental to "
            "ring generalship and controlling the pace of a fight."
        ),
        "Footwork and Ring Generalship": (
            "Movement patterns that control distance, create angles, and enable offence and defence. "
            "Basic footwork: step-drag (maintain stance width), lateral movement (circle away from "
            "power hand), pivot (create angles), and cutting angles (reduce opponent's retreat "
            "space). Never cross your feet. Always move the foot in the direction you intend to go "
            "first. Circle away from the opponent's power hand to avoid their strongest punch."
        ),
    },
    "Combinations": {
        "1-2 (Jab-Cross)": (
            "The most fundamental combination in boxing. The jab occupies the opponent's guard "
            "and establishes range; the cross follows immediately to the open target. The 1-2 "
            "must be practised until it becomes reflexive. Variations: 1-2 to the body, 1-2 to "
            "the head followed by a body shot."
        ),
        "1-2-3 (Jab-Cross-Lead Hook)": (
            "One of the most common three-punch combinations. The cross closes distance "
            "while the lead hook targets the exposed chin. The hook can go to the head or body. "
            "Key: after the cross, do not return the rear hand all the way to guard — let it "
            "flow directly into the hook setup."
        ),
        "1-2-3-2 (Jab-Cross-Hook-Cross)": (
            "A four-punch combination ending with the rear power hand. The first three punches "
            "break down the defence; the final cross lands on a partially open target. Common "
            "at high levels of boxing."
        ),
        "1-6 (Jab to head, hook to body)": (
            "Level-change combination: the jab draws the guard high, then the body hook scores "
            "on the now-exposed midsection. Forces the opponent to make a choice."
        ),
        "2-3-2 (Cross-Hook-Cross)": (
            "A power combination starting with the cross. Effective when the opponent is already "
            "stunned or when initiating from a right-hand range. The hook turns the head, "
            "presenting the chin for the finishing cross."
        ),
        "Double Jab-Cross": (
            "Two quick jabs followed by the cross. The first jab probes, the second jab draws a "
            "reaction (opponent may parry or flinch), and the cross follows through the momentary "
            "opening created by the opponent's reaction."
        ),
        "Jab to Body-Cross to Head": (
            "A fundamental level-change combination. The body jab forces the elbows to drop; "
            "the cross targets the now-exposed chin. Telegraphing is minimised because both "
            "punches look the same at the start."
        ),
    },
    "Stances and Footwork": {
        "Orthodox Stance": (
            "Standard stance for right-handed fighters: left foot forward, right foot back, "
            "weight approximately 60% front / 40% rear. The lead left hand jabs; the rear right "
            "hand delivers power shots. Feet are shoulder-width apart at 45-degree angle to the "
            "opponent. Lead toe points toward the opponent; rear foot is slightly turned out."
        ),
        "Southpaw Stance": (
            "Mirrored stance for left-handed fighters: right foot forward, left foot back. "
            "In orthodox vs southpaw matchups, both fighters' power hands are on the outside — "
            "creating unique angle opportunities. The southpaw's cross (right hand) can land "
            "over the orthodox fighter's jab. Footwork advantage: each fighter tries to get their "
            "lead foot to the outside of the opponent's lead foot."
        ),
        "Outside Foot Position": (
            "Positioning your lead foot to the outside of the opponent's lead foot. This gives "
            "you an angular advantage: your power hand has a clear line to their head, while "
            "their power hand must reach across to hit you."
        ),
        "Cutting Off the Ring": (
            "Moving to limit the opponent's retreat options by stepping to angles that reduce "
            "available space, forcing them into a corner or toward the ropes. A skilled pressure "
            "fighter uses lateral steps to herd the opponent into a confined area."
        ),
        "Step-Drag": (
            "The foundational boxing movement: the foot in the direction of movement steps "
            "first, the other foot drags to restore stance width. Moving forward: step lead, "
            "drag rear. Moving back: step rear, drag lead. This maintains balance throughout."
        ),
    },
    "Training Methods": {
        "Shadowboxing": (
            "Boxing an imaginary opponent to develop technique, footwork, and fight visualisation. "
            "Effective shadowboxing requires: a clear mental image of a specific opponent, "
            "working on specific game plan elements, realistic movement (not posing), and "
            "maintaining full technique throughout. Shadowboxing can be done for rounds (3-5 "
            "min) or as part of a warm-up. It is the safest way to develop motor patterns."
        ),
        "Heavy Bag Work": (
            "Hitting a heavy punching bag to develop power, conditioning, and combinations. "
            "Effective heavy bag training uses full footwork (circle, pivot, move), realistic "
            "combinations (not just power shots), and defence between combinations. "
            "Sessions: 3-6 rounds of 2-3 minutes with 1-minute rest."
        ),
        "Speed Bag": (
            "A small teardrop-shaped bag on a rebound platform that develops hand-eye "
            "coordination, hand speed, rhythm, and shoulder endurance. Hit with a circular "
            "fist motion, alternating hands. Allow the bag to rebound three times before the "
            "next strike. Start slow; speed develops naturally."
        ),
        "Focus Mitts (Pads)": (
            "Training with a partner holding focus mitts, allowing real-time feedback on "
            "technique, timing, and combinations. Mitt work bridges shadowboxing and sparring. "
            "A skilled holder simulates realistic offensive and defensive patterns and calls "
            "combinations or throws punches for the fighter to slip and counter."
        ),
        "Sparring": (
            "Controlled fighting practice with a partner, using protective equipment. Sparring "
            "is irreplaceable for developing ring intelligence, timing, pressure tolerance, "
            "and the ability to apply techniques against a resisting opponent. Types: "
            "technical (light contact), hard (near-full intensity), and specific (one partner "
            "works offence only). Key principle: spar smart — accumulated damage from sparring "
            "is the number one reason fighters decline prematurely."
        ),
        "Roadwork": (
            "Running and outdoor conditioning builds the aerobic base, leg endurance, and mental "
            "toughness. Long runs: 30-60 min at easy pace. Interval work: 200-400m sprints with "
            "full recovery. Most boxing coaches schedule roadwork in the morning before gym."
        ),
        "Jump Rope": (
            "Skipping rope develops footwork coordination, rhythm, cardiovascular conditioning, "
            "calf strength, and timing. Patterns: two-foot bounce, alternating footstep "
            "(simulates running), boxer step (side to side), and high knees. "
            "Sessions: 10-15 minutes continuous or in timed rounds."
        ),
        "Double-End Bag": (
            "Attached top and bottom with elastic cords, the double-end bag recoils at the "
            "fighter on contact, developing timing, reflexes, and accuracy. Unlike the heavy bag, "
            "it demands rhythm and precision — it punishes poor timing by bouncing back "
            "unpredictably. Excellent for developing combination flow and evasion."
        ),
    },
    "Physical Conditioning": {
        "Periodisation": (
            "Systematic planning of training intensity and volume over a training cycle. A boxing "
            "camp typically follows: General Prep Phase (8-12 weeks out) — aerobic base, "
            "strength; Specific Prep Phase (4-8 weeks) — sport-specific conditioning, sparring; "
            "Peak Phase (1-3 weeks) — reduce volume, maintain intensity, taper. Never try to "
            "get significantly fitter in the final week: rest and absorb the training."
        ),
        "Aerobic Base": (
            "The aerobic energy system powers sustained effort over rounds. Built through long "
            "slow distance work at a conversational pace (heart rate 60-70% max). A strong "
            "aerobic base accelerates recovery between combinations, between rounds, and "
            "between training sessions."
        ),
        "Anaerobic Conditioning": (
            "High-intensity energy system used during explosive exchanges. Built through sprint "
            "intervals, circuit training, and hard sparring. Anaerobic capacity determines how "
            "hard you can work in short bursts and how quickly you recover from them."
        ),
        "Weight Cutting": (
            "Reducing body weight before a weigh-in through water loss and food restriction. "
            "Safe cutting: no more than 5-8% of body weight. More impairs performance, "
            "recovery, and long-term health. Same-day weigh-ins are increasingly required "
            "to reduce dangerous cutting."
        ),
        "Strength Training for Boxers": (
            "Compound movements build the kinetic chain power underlying punching force: "
            "deadlift, squat, overhead press, power clean. Phase 1: maximal strength "
            "(3-5 reps, heavy). Phase 2: power conversion (explosive movements — med ball "
            "throws, plyometrics, Olympic lifts). Taper heavy lifting 6-8 weeks before a fight."
        ),
    },
}

COACHING_QA_PAIRS = [
    {
        "q": "How do I improve my jab?",
        "a": (
            "The jab is the foundation of boxing — every improvement here pays dividends across your "
            "entire game.\n\n"
            "1. EXTENSION: Fully extend the jab. Many fighters stop 80% of the way. At full extension, "
            "your fist rotates palm-down and your shoulder shrugs slightly to protect your chin.\n\n"
            "2. SNAP: The power of the jab comes from its snap, not push. Throw it out and pull it back "
            "immediately. Practise on the speed bag to develop this rhythm.\n\n"
            "3. HIP ENGAGEMENT: Even the jab involves a subtle hip rotation. Push off your rear foot "
            "and let that energy flow through your hip and into the punch.\n\n"
            "4. RANGE: You should be at jab range — where you can just touch the opponent with a fully "
            "extended jab. Most fighters work too close.\n\n"
            "5. DECEPTION: Use the jab in combinations, as a range finder, to the body, doubled up, "
            "and as a feint. A predictable jab is easily parried.\n\n"
            "Drills: 100 jabs on the heavy bag per session (focus on form). 5 rounds of shadowboxing "
            "using only the jab. Mitt work where your coach calls 'jab' randomly."
        )
    },
    {
        "q": "What is the best way to defend against a southpaw?",
        "a": (
            "Fighting a southpaw is a tactical puzzle many orthodox fighters never solve.\n\n"
            "FOOTWORK FIRST: The key battle is foot position. You want your lead foot (left) on the "
            "OUTSIDE of their lead foot (right). This gives you a clear line to their head with your "
            "right hand, while their left hand must reach across to hit you. Circle LEFT (to their "
            "right) to move off their left hand.\n\n"
            "JAB TO THE BODY: Southpaws expose their right side when they jab. A jab to their body "
            "is frequently open and hard for them to see.\n\n"
            "RIGHT HAND OVER THEIR JAB: When the southpaw jabs, their right side opens. Your cross "
            "can catch them coming in.\n\n"
            "LEFT HOOK TO THE BODY: Their liver is exposed. A short left hook to the body is a "
            "devastating weapon against any southpaw.\n\n"
            "AVOID CIRCLING RIGHT: Moving right takes you directly into their left (power) hand."
        )
    },
    {
        "q": "How do I stop getting hit by the counter right hand?",
        "a": (
            "Getting countered after your jab is one of the most common problems in boxing.\n\n"
            "1. SLIP THE JAB: When you throw the jab, slip your head slightly right as the jab "
            "extends. If their counter right comes, it misses because your head has moved.\n\n"
            "2. PULL THE JAB BACK FASTER: The hand should return to guard as fast as it extended. "
            "A slow retraction is an invitation to counter.\n\n"
            "3. JAB FROM DIFFERENT ANGLES: Mix in jabs to the body, at different distances, while "
            "moving laterally. Predictable jabs are countered; unpredictable ones are not.\n\n"
            "4. FOLLOW THE JAB: The best defence is not giving them time to counter. If your jab "
            "lands, immediately follow with the cross (the 1-2).\n\n"
            "5. LEAD WITH THE SHOULDER: Slightly rolling the shoulder forward with the jab protects "
            "the chin and narrows the target."
        )
    },
    {
        "q": "How do I develop knockout power?",
        "a": (
            "Knockout power comes from mechanics, genetics, timing, and conditioning.\n\n"
            "MECHANICS (most trainable):\n"
            "- Hip rotation is the primary source of power. Every hard punch rotates from the feet up: "
            "foot pivot, knee drive, hip rotation, shoulder rotation, arm extension.\n"
            "- Relax before impact: tense muscles slow the punch. Stay relaxed and contract at the "
            "moment of impact (like cracking a whip).\n"
            "- Land with the first two knuckles (index and middle finger).\n\n"
            "TIMING:\n"
            "- Punching an opponent who is moving INTO your punch doubles the effective force. "
            "This is why counters produce knockouts.\n\n"
            "CONDITIONING:\n"
            "- Core rotation: medicine ball rotational throws, cable rotations.\n"
            "- Hip power: deadlifts, squats, kettlebell swings.\n\n"
            "Fix mechanics first. Poor mechanics prevent every fighter from reaching their ceiling."
        )
    },
    {
        "q": "What is the best training schedule for an amateur boxer preparing for a tournament?",
        "a": (
            "12-week tournament preparation:\n\n"
            "WEEKS 1-4 (General Preparation):\n"
            "Roadwork: 4-5 mornings/week, 30-40 min easy run + 6x200m sprints twice/week.\n"
            "Gym: 5 sessions — shadowboxing, heavy bag, mitts, speed bag, double-end bag, light sparring.\n"
            "Strength: 2x/week compound lifts (squat, deadlift, press).\n\n"
            "WEEKS 5-8 (Specific Preparation):\n"
            "Increase sparring to 6-8 rounds, 3x/week. Sprint work replaces easy running.\n"
            "Reduce heavy lifting — switch to explosive (med ball, plyometrics).\n\n"
            "WEEKS 9-10 (Peak):\n"
            "Maintain intensity, reduce volume 20%. Sparring: 2x/week, sharp.\n\n"
            "WEEKS 11-12 (Taper):\n"
            "Light technical work only. No new techniques. Trust the preparation.\n"
            "Sleep 8-9 hours. You cannot get fitter in the final week — only more tired."
        )
    },
    {
        "q": "How do I improve head movement in boxing?",
        "a": (
            "Head movement must be trained systematically.\n\n"
            "THE KEY: Head movement originates from the knees and hips, not the neck. Bend your "
            "knees and you move your entire body. Bend only your neck and you move a few inches.\n\n"
            "THE FOUR CORE MOVEMENTS:\n"
            "1. Slip outside (torso rotates, head moves outside the punch)\n"
            "2. Slip inside (head moves inside — riskier, creates hook/uppercut counters)\n"
            "3. Pull back (lean away — simplest, useful from range)\n"
            "4. Roll under (U-shape under hooks — knees not neck)\n\n"
            "TRAINING:\n"
            "- Slip rope: hang a rope at head height, practise slipping side to side.\n"
            "- Double-end bag: forces head movement or you get hit on the return.\n"
            "- Dedicated sparring: partner only throws jabs, you only slip and counter.\n\n"
            "ELITE HABIT: Move your head BEFORE and DURING your own punches. Every combination "
            "should end with a defensive movement — not standing still."
        )
    },
    {
        "q": "What is the liver shot and how do I set it up?",
        "a": (
            "The liver shot is one of the most devastating punches in boxing.\n\n"
            "ANATOMY: The liver is under the right side of the ribcage (your left side when facing "
            "the opponent). When struck cleanly, it causes intense wave-like pain that is nearly "
            "impossible to fight through.\n\n"
            "THE PUNCH: A left hook to the body, thrown slightly upward to get under the elbow and "
            "rib cage.\n\n"
            "SETUPS:\n"
            "1. 1-2-3 to body: Jab head, cross head, left hook BODY. First two punches raise the guard.\n\n"
            "2. FEINT HIGH: Fake a jab to the head. As the guard comes up, attack the exposed body.\n\n"
            "3. ACCUMULATE: Throw body shots consistently from round 1. The liver bruises "
            "progressively — a tap in round 8 hurts more than a hard shot in round 1.\n\n"
            "4. CATCH THE CROSS: When they rotate into their cross, their right side opens. "
            "Time a short left hook to the body to catch the rotation.\n\n"
            "Never telegraph the body shot. The setup punch must genuinely threaten the head."
        )
    },
    {
        "q": "How do I use the jab to control distance?",
        "a": (
            "Distance management is the jab's most important role beyond scoring.\n\n"
            "THE RANGE CONCEPT: You want to be at a distance where your fully extended jab just "
            "reaches the opponent's face. At this range you can reach them; they cannot reach "
            "you without stepping in first — giving you time to react.\n\n"
            "USING THE JAB TO ESTABLISH RANGE:\n"
            "- Use a measuring jab (pawing jab) to gauge distance as you enter and exit.\n"
            "- If the opponent is too close, push the jab into them to create space while "
            "simultaneously stepping back with the rear foot.\n"
            "- If the opponent circles away, step-drag to close distance, then fire the jab.\n\n"
            "CONTROL PATTERNS:\n"
            "- Double jab + reset: two quick jabs then step back to reset range.\n"
            "- Jab + pivot: jab and pivot 45 degrees to a new angle. They must reset; you don't.\n"
            "- Body jab: shifts their guard down, opens the head, and tests their reaction.\n\n"
            "KEY: The opponent should always be reacting to your jab. If they are setting the pace, "
            "your jab is not working hard enough."
        )
    },
    {
        "q": "How should I warm up before sparring?",
        "a": (
            "A proper warm-up before sparring reduces injury risk and improves performance.\n\n"
            "PHASE 1 — GENERAL WARM-UP (10-15 min):\n"
            "3 rounds of jump rope at easy to moderate pace. This raises core temperature and "
            "activates the cardiovascular system.\n\n"
            "PHASE 2 — MOBILITY (5-10 min):\n"
            "Neck rolls, shoulder circles, hip circles, knee circles, ankle rolls. "
            "Dynamic stretches: leg swings, arm circles, torso rotations. "
            "Do NOT do static stretching before sparring — it reduces explosive output.\n\n"
            "PHASE 3 — BOXING-SPECIFIC ACTIVATION (10-15 min):\n"
            "2 rounds of easy shadowboxing focusing on footwork and defence. "
            "1-2 rounds of light bag work to groove technique. "
            "Brief mitt work to sharpen reflexes.\n\n"
            "PHASE 4 — MENTAL PREPARATION:\n"
            "Review your game plan. Identify the specific technique or tactic you will work on "
            "in this sparring session. Purposeful sparring beats random sparring every time.\n\n"
            "NOTE: Spar within 15-20 minutes of completing the warm-up. If you wait longer, "
            "your body will cool down and you have to warm up again."
        )
    },
]

TRAINING_PROGRAMS = [
    {
        "title": "12-Week Boxing Conditioning Cycle",
        "content": (
            "PHASE 1 (Weeks 1-4): AEROBIC BASE BUILDING\n"
            "Goal: Build the cardiovascular engine that will power all future high-intensity work.\n"
            "Roadwork: 5 mornings/week. 3 sessions of 40-min easy runs. 2 sessions of "
            "20-min tempo runs (comfortably hard).\n"
            "Gym: 3 sessions/week. 3 rounds shadowboxing, 4 rounds heavy bag (moderate), "
            "4 rounds mitts, 2 rounds skip rope, 2 rounds speed bag. Focus on TECHNIQUE.\n"
            "Strength: 2x/week. Squats 4x6, Romanian deadlift 3x8, press 3x8, pull-ups 3x max.\n\n"

            "PHASE 2 (Weeks 5-8): SPECIFIC CONDITIONING\n"
            "Goal: Convert aerobic base into boxing-specific power-endurance.\n"
            "Roadwork: 4 mornings/week. 2 steady runs. 2 sprint sessions: 10x100m or hill sprints.\n"
            "Gym: 4 sessions/week. Heavy bag (hard, combinations), mitts (competition simulation), "
            "3 rounds sparring (technical), double-end bag.\n"
            "Strength: Shift to power — jump squats, kettlebell swings, med ball throws.\n\n"

            "PHASE 3 (Weeks 9-12): PEAK AND TAPER\n"
            "Week 9: Maintain intensity, reduce volume 15%. Sparring up to 6 rounds.\n"
            "Week 10: Reduce volume further. 1 hard spar/week, 2 technical.\n"
            "Week 11: Volume down 40% from peak. Technique work only. Rest.\n"
            "Week 12: Light movement, bag work only. Sleep 9 hours. Trust the preparation.\n"
        )
    },
    {
        "title": "Daily Boxing Gym Session Structure",
        "content": (
            "A well-structured boxing session runs 90-120 minutes.\n\n"
            "WARM-UP (15-20 min): 5-10 min jump rope. Joint mobility. 2 rounds easy shadowboxing.\n\n"
            "MAIN TRAINING (60-80 min):\n"
            "3-4 rounds shadowboxing (specific combinations, footwork, game plan).\n"
            "4-6 rounds heavy bag (first 2 moderate, last 2-4 hard).\n"
            "4-6 rounds mitts (combinations, reactions, defensive drills).\n"
            "3-6 rounds sparring (technical or hard, depending on phase).\n"
            "2 rounds speed bag. 2 rounds double-end bag.\n\n"
            "CONDITIONING (10-15 min): Core circuit — planks, rotational crunches, ab wheel.\n"
            "Neck bridging (important injury prevention — build up gradually).\n\n"
            "COOL-DOWN (10 min): Easy shadowboxing. Full-body stretching.\n"
        )
    },
    {
        "title": "Beginner Boxing Programme (First 3 Months)",
        "content": (
            "MONTH 1: FOUNDATIONS\n"
            "Focus: stance, guard, jab, basic footwork.\n"
            "3 sessions/week. Each session: 15 min jump rope, 3 rounds shadowboxing "
            "(jab only, footwork), 3 rounds heavy bag (jab only), 1 round speed bag.\n"
            "No sparring. Learn to move without crossing feet.\n\n"

            "MONTH 2: COMBINATIONS\n"
            "Focus: 1-2, 1-2-3, body shots, basic defence (slip, roll).\n"
            "4 sessions/week. Add 3 rounds mitt work. Begin controlled technical sparring "
            "(last 2 weeks only, light contact, focus on jab and movement).\n\n"

            "MONTH 3: RING CRAFT\n"
            "Focus: distance management, angles, defensive combinations.\n"
            "4-5 sessions/week. Increase sparring to 2-3 rounds/session. "
            "Begin studying your own technique on video. Add roadwork 3 mornings/week.\n\n"

            "KEY PRINCIPLE: Master the jab before everything else. A sharp, deceptive jab "
            "makes every other technique easier. Do not rush to advanced combinations."
        )
    },
]


def generate_glossary(output_dir: Path) -> int:
    """Write boxing glossary, coaching Q&A, and training programs to files."""
    gen_dir = output_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Glossary
    lines = [
        "BOXING COACHING GLOSSARY\n",
        "A comprehensive reference for fighters, coaches, and students of boxing.\n\n",
    ]
    for cat, terms in BOXING_GLOSSARY.items():
        lines.append(f"{cat.upper()}\n{'─'*60}\n\n")
        for term, defn in terms.items():
            lines.append(f"{term}\n{defn}\n\n")
    text = '\n'.join(lines)
    (gen_dir / "boxing_glossary.txt").write_text(text, encoding='utf-8')
    print(f"[Generated] boxing_glossary.txt ({len(text):,} chars)")

    # Q&A
    qa_lines = ["BOXING COACHING Q&A\n\n"]
    for pair in COACHING_QA_PAIRS:
        qa_lines.append(f"Q: {pair['q']}\n\nA: {pair['a']}\n\n{'─'*60}\n\n")
    qa_text = '\n'.join(qa_lines)
    (gen_dir / "coaching_qa.txt").write_text(qa_text, encoding='utf-8')
    print(f"[Generated] coaching_qa.txt ({len(qa_text):,} chars)")

    # Training programs
    prog_lines = ["BOXING TRAINING PROGRAMMES\n\n"]
    for p in TRAINING_PROGRAMS:
        prog_lines.append(f"# {p['title']}\n\n{p['content']}\n\n{'─'*60}\n\n")
    prog_text = '\n'.join(prog_lines)
    (gen_dir / "training_programs.txt").write_text(prog_text, encoding='utf-8')
    print(f"[Generated] training_programs.txt ({len(prog_text):,} chars)")

    return 1


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════

def print_summary(raw_dir: Path) -> None:
    """Print a summary of all collected data."""
    txt_files = list(raw_dir.glob('**/*.txt'))
    total_chars = 0
    for f in txt_files:
        try:
            total_chars += len(f.read_text(encoding='utf-8', errors='replace'))
        except Exception:
            pass

    by_src: Dict[str, int] = {}
    for f in txt_files:
        parts = f.relative_to(raw_dir).parts
        src = parts[0] if parts else 'other'
        by_src[src] = by_src.get(src, 0) + 1

    print(f"\n{'='*55}")
    print(f"  DATA COLLECTION SUMMARY")
    print(f"{'='*55}")
    print(f"  Total .txt files : {len(txt_files):,}")
    print(f"  Total characters : {total_chars:,} ({total_chars/1e6:.1f}M)\n")
    print(f"  By source:")
    for src, cnt in sorted(by_src.items(), key=lambda x: -x[1]):
        print(f"    {src:<22} {cnt:>5} files")

    print()
    if total_chars < 1_000_000:
        print("  Status: ⚠  Very little data — model will overfit")
    elif total_chars < 5_000_000:
        print("  Status: 🟡 Small — good for pipeline testing")
    elif total_chars < 20_000_000:
        print("  Status: 🟢 Good — should produce reasonable results")
    else:
        print("  Status: ✅ Excellent — full training recommended")
    print(f"\n  Next: python scripts/prepare_data.py")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BoxingGPT Data Collector v3 — boxing-focused, clean output"
    )
    parser.add_argument('--all',            action='store_true')
    parser.add_argument('--wikipedia',      action='store_true')
    parser.add_argument('--youtube',        action='store_true')
    parser.add_argument('--reddit',         action='store_true')
    parser.add_argument('--stackexchange',  action='store_true')
    parser.add_argument('--glossary',       action='store_true')
    parser.add_argument('--output_dir',     type=str,   default='data/raw/')
    parser.add_argument('--delay',          type=float, default=1.0,
                        help="Base request delay in seconds")
    parser.add_argument('--max_videos',     type=int,   default=500)
    parser.add_argument('--max_posts',      type=int,   default=150)
    parser.add_argument('--max_questions',  type=int,   default=100)
    args = parser.parse_args()

    raw_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_all = args.all or not any([
        args.wikipedia, args.youtube, args.reddit,
        args.stackexchange, args.glossary,
    ])

    print(f"\n🥊 BoxingGPT Data Collector v3")
    print(f"   Output: {raw_dir.resolve()}\n")

    if run_all or args.glossary:
        generate_glossary(raw_dir)

    if run_all or args.wikipedia:
        collect_wikipedia(raw_dir, delay=args.delay)

    if run_all or args.youtube:
        collect_youtube_transcripts(raw_dir, max_videos=args.max_videos)

    if run_all or args.reddit:
        collect_reddit(raw_dir, posts_per_sub=args.max_posts, delay=max(args.delay * 2, 2.5))

    if run_all or args.stackexchange:
        collect_stackexchange(raw_dir, max_per_tag=args.max_questions, delay=args.delay)

    print_summary(raw_dir)


if __name__ == '__main__':
    main()

