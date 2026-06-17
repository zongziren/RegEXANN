"""
gen_query.py — RegExANN query + ground-truth generator
=======================================================
Regex styles follow the paper (RegExANN, SIGMOD'27):
  substring   →  neural
  prefix      →  ^deep
  suffix      →  search$
  alternation →  graph|network
  wildcard    →  deep.*learning
  mixed       →  ann.*gpu|gpu.*ann

每条生成的查询都保证在字符串语料库中至少有 1 个匹配结果（case-insensitive）。
"""

import numpy as np
import re
import random
import argparse
from tqdm import tqdm
from typing import Optional, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_fvecs(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        dim = int.from_bytes(f.read(4), byteorder="little")
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, dim + 1)
        return data[:, 1:]


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ─────────────────────────────────────────────────────────────────────────────
# Token extraction — shared by all generators
# ─────────────────────────────────────────────────────────────────────────────

def extract_words(title: str, min_len: int = 3) -> List[str]:
    """Return lower-case alphabetic tokens of length >= min_len."""
    return [w.lower() for w in re.findall(r'[a-zA-Z]+', title) if len(w) >= min_len]


# ─────────────────────────────────────────────────────────────────────────────
# Per-style regex builders
# Each returns a regex *string* (not compiled) or None on failure.
# ─────────────────────────────────────────────────────────────────────────────

def build_substring(words: List[str]) -> Optional[str]:
    """Single meaningful token — broadest match."""
    if not words:
        return None
    return random.choice(words)


def build_prefix(words: List[str]) -> Optional[str]:
    """Anchor the first token to start-of-string: ^token"""
    if not words:
        return None
    return f"^{words[0]}"


def build_suffix(words: List[str]) -> Optional[str]:
    """Anchor the last token to end-of-string: token$"""
    if not words:
        return None
    return f"{words[-1]}$"


def build_alternation(words: List[str]) -> Optional[str]:
    """Two distinct tokens joined by |:  token1|token2"""
    if len(words) < 2:
        return None
    w1, w2 = random.sample(words, 2)
    return f"{w1}|{w2}"


def build_wildcard(words: List[str]) -> Optional[str]:
    """Two tokens with .* between them:  token1.*token2"""
    if len(words) < 2:
        return None
    # Pick two tokens that appear in the title in order (indices i < j)
    indices = random.sample(range(len(words)), 2)
    i, j = min(indices), max(indices)
    return f"{words[i]}.*{words[j]}"


def build_mixed(words: List[str]) -> Optional[str]:
    """Bidirectional wildcard alternation:  a.*b|b.*a"""
    if len(words) < 2:
        return None
    indices = random.sample(range(len(words)), 2)
    i, j = min(indices), max(indices)
    a, b = words[i], words[j]
    return f"{a}.*{b}|{b}.*{a}"


STYLE_BUILDERS = {
    "substring":   build_substring,
    "prefix":      build_prefix,
    "suffix":      build_suffix,
    "alternation": build_alternation,
    "wildcard":    build_wildcard,
    "mixed":       build_mixed,
}

STYLES = list(STYLE_BUILDERS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Core: generate one valid (regex, candidates) pair
# ─────────────────────────────────────────────────────────────────────────────

def try_build_regex(title: str, style: str) -> Optional[str]:
    """Attempt to build a regex of the given style from `title`."""
    words = extract_words(title)
    builder = STYLE_BUILDERS[style]
    return builder(words)


def find_candidates(pattern: re.Pattern, titles: List[str]) -> List[int]:
    return [i for i, t in enumerate(titles) if pattern.search(t)]


def generate_one_query(
    title_idx: int,
    titles: list[str],
    style: str,
    max_attempts: int = 20,
) -> Tuple[Optional[str], Optional[List[int]]]:
    """
    Try up to `max_attempts` times to produce a regex (from nearby titles)
    that matches at least 1 string in the corpus.

    Strategy:
      1. First try the source title itself.
      2. If that fails, try random titles from the corpus.
      3. For each candidate title, attempt all styles (shuffled) until one works.
    Returns (regex_str, matching_indices) or (None, None) on total failure.
    """
    source_title = titles[title_idx]

    # Pool: source title first, then random others
    pool = [source_title]
    pool += [titles[random.randint(0, len(titles) - 1)] for _ in range(max_attempts - 1)]

    styles_to_try = [style] + [s for s in STYLES if s != style]

    for attempt_title in pool:
        for s in styles_to_try:
            rx_str = try_build_regex(attempt_title, s)
            if rx_str is None:
                continue
            try:
                pattern = re.compile(rx_str, re.IGNORECASE)
            except re.error:
                continue
            candidates = find_candidates(pattern, titles)
            if candidates:
                return rx_str, candidates
            # Pattern compiled but matched nothing — try next style

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_queries_and_groundtruth(
    title_file: str,
    fvec_file: str,
    query_out: str,
    gt_out: str,
    n_queries: int = 100,
    topk: int = 10,
    style: str = "mixed",
    seed: int = 42,
    max_attempts: int = 30,
):
    random.seed(seed)
    np.random.seed(seed)

    # ── Load data ────────────────────────────────────────────────────────────
    titles = [line.strip() for line in open(title_file, encoding="utf-8")]
    vectors = read_fvecs(fvec_file)
    assert len(titles) == len(vectors), (
        f"Mismatch: {len(titles)} titles vs {len(vectors)} vectors"
    )
    N = len(titles)
    print(f"[INFO] Loaded {N} strings and vectors (dim={vectors.shape[1]})")

    # ── Sample source indices (deduplicated, shuffled) ───────────────────────
    source_pool = list(range(N))
    random.shuffle(source_pool)
    source_iter = iter(source_pool)

    written = 0
    skipped = 0

    with open(query_out, "w", encoding="utf-8") as fq, \
         open(gt_out, "w", encoding="utf-8") as fg:

        pbar = tqdm(total=n_queries, desc=f"Generating queries [{style}]")

        while written < n_queries:
            try:
                qid = next(source_iter)
            except StopIteration:
                print(f"[WARN] Exhausted all {N} source titles after {written} queries.")
                break

            query_vec = vectors[qid]

            rx_str, candidates = generate_one_query(
                qid, titles, style, max_attempts=max_attempts
            )

            if rx_str is None or not candidates:
                skipped += 1
                continue

            # ── Ground truth: top-K by exact Euclidean distance ───────────
            cand_vecs = vectors[candidates]           # (M, dim)
            diffs = cand_vecs - query_vec             # broadcast
            dists = np.linalg.norm(diffs, axis=1)
            order = np.argsort(dists)[:topk]
            top_k_ids = [candidates[i] for i in order]

            # ── Write query line: <regex> <v0> <v1> ... <vd-1> ───────────
            vec_str = " ".join(f"{x:.6f}" for x in query_vec)
            fq.write(f"{rx_str} {vec_str}\n")

            # ── Write ground-truth line: space-separated IDs ─────────────
            fg.write(" ".join(str(i) for i in top_k_ids) + "\n")

            written += 1
            pbar.update(1)

        pbar.close()

    print(f"[INFO] Wrote {written} queries to '{query_out}'")
    print(f"[INFO] Wrote {written} ground-truth lines to '{gt_out}'")
    if skipped:
        print(f"[WARN] Skipped {skipped} source titles (no valid regex found)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RegExANN queries and ground-truth (paper-compliant)"
    )
    parser.add_argument("title_file", help="Path to strings .txt file (one per line)")
    parser.add_argument("fvec_file",  help="Path to .fvecs vector file")
    parser.add_argument("query_out",  help="Output query file path")
    parser.add_argument("gt_out",     help="Output ground-truth file path")
    parser.add_argument("--n_queries",    type=int,   default=100,
                        help="Number of queries to generate (default: 100)")
    parser.add_argument("--topk",         type=int,   default=10,
                        help="Top-K ground-truth results per query (default: 10)")
    parser.add_argument("--style",        type=str,   default="mixed",
                        choices=STYLES + ["all"],
                        help="Regex style (default: mixed). Use 'all' to cycle evenly.")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max_attempts", type=int,   default=30,
                        help="Max title attempts per query before skipping (default: 30)")
    args = parser.parse_args()

    # 'all' → cycle through styles evenly
    if args.style == "all":
        # We monkey-patch: override generate_one_query to rotate styles
        _style_cycle = [STYLES[i % len(STYLES)] for i in range(args.n_queries)]
        _orig_gen = generate_one_query

        def _cycling_gen(title_idx, titles, style, max_attempts=30):
            # style argument is ignored; we pop from cycle
            s = _style_cycle[title_idx % len(_style_cycle)]
            return _orig_gen(title_idx, titles, s, max_attempts)

        import builtins
        generate_one_query = _cycling_gen  # noqa: F811

    generate_queries_and_groundtruth(
        title_file=args.title_file,
        fvec_file=args.fvec_file,
        query_out=args.query_out,
        gt_out=args.gt_out,
        n_queries=args.n_queries,
        topk=args.topk,
        style=args.style if args.style != "all" else "mixed",
        seed=args.seed,
        max_attempts=args.max_attempts,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Usage examples
# ─────────────────────────────────────────────────────────────────────────────
# python gen_query.py ./sift/sift_titles_clean.txt ./sift/sift_vectors.fvecs \
#     ./sift/query.txt ./sift/groundtruth.txt --n_queries 1000 --topk 10 --style mixed
#

# python gen_query.py ./sift/sift_titles_clean.txt ./sift/sift_vectors.fvecs \
#     ./sift/query.txt ./sift/groundtruth.txt --n_queries 1000 --topk 10 --style all