"""
gen_query.py — RegExANN query + ground-truth generator
=======================================================
Regex styles follow the paper (RegExANN, SIGMOD'27):
  substring   ->  neural
  prefix      ->  ^deep
  suffix      ->  search$
  alternation ->  graph|network
  wildcard    ->  deep.*learning
  mixed       ->  ann.*gpu|gpu.*ann

Query file format:
  <regex> <v0> <v1> ... <vd-1>

Groundtruth file format:
  <id0> <id1> ... <id{k-1}>

Each generated query is guaranteed to have regex selectivity within:
  [min_selectivity, max_selectivity]

Default:
  min_selectivity = 0.01  # 1%
  max_selectivity = 0.10  # 10%
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
    return [
        w.lower()
        for w in re.findall(r"[a-zA-Z]+", title)
        if len(w) >= min_len
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Per-style regex builders
# Each returns a regex string or None on failure.
# ─────────────────────────────────────────────────────────────────────────────

def build_substring(words: List[str]) -> Optional[str]:
    """Single meaningful token."""
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
    """Two distinct tokens joined by |: token1|token2"""
    if len(words) < 2:
        return None
    w1, w2 = random.sample(words, 2)
    return f"{w1}|{w2}"


def build_wildcard(words: List[str]) -> Optional[str]:
    """Two tokens with .* between them: token1.*token2"""
    if len(words) < 2:
        return None

    indices = random.sample(range(len(words)), 2)
    i, j = min(indices), max(indices)

    if words[i] == words[j]:
        return None

    return f"{words[i]}.*{words[j]}"


def build_mixed(words: List[str]) -> Optional[str]:
    """Bidirectional wildcard alternation: a.*b|b.*a"""
    if len(words) < 2:
        return None

    indices = random.sample(range(len(words)), 2)
    i, j = min(indices), max(indices)

    a, b = words[i], words[j]

    if a == b:
        return None

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
# Regex matching helpers
# ─────────────────────────────────────────────────────────────────────────────

def try_build_regex(title: str, style: str) -> Optional[str]:
    """Attempt to build a regex of the given style from title."""
    words = extract_words(title)
    builder = STYLE_BUILDERS[style]
    return builder(words)


def find_candidates(pattern: re.Pattern, titles: List[str]) -> List[int]:
    return [i for i, t in enumerate(titles) if pattern.search(t)]


def generate_one_query(
    title_idx: int,
    titles: List[str],
    style: str,
    min_matches: int,
    max_matches: int,
    max_attempts: int = 30,
) -> Tuple[Optional[str], Optional[List[int]]]:
    """
    Generate one valid regex query whose selectivity is within:

        min_matches <= #matched_strings <= max_matches

    Strategy:
      1. First try the source title itself.
      2. Then try random titles from the corpus.
      3. For each attempted title, try the target style first.
      4. If that fails, try the remaining styles.
    """
    source_title = titles[title_idx]

    pool = [source_title]
    pool += [
        titles[random.randint(0, len(titles) - 1)]
        for _ in range(max_attempts - 1)
    ]

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
            m = len(candidates)

            if min_matches <= m <= max_matches:
                return rx_str, candidates

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
    min_selectivity: float = 0.01,
    max_selectivity: float = 0.10,
):
    random.seed(seed)
    np.random.seed(seed)

    titles = [line.strip() for line in open(title_file, encoding="utf-8")]
    vectors = read_fvecs(fvec_file)

    assert len(titles) == len(vectors), (
        f"Mismatch: {len(titles)} titles vs {len(vectors)} vectors"
    )

    N = len(titles)
    dim = vectors.shape[1]

    print(f"[INFO] Loaded {N} strings and vectors (dim={dim})")

    min_matches = max(topk, int(np.ceil(N * min_selectivity)))
    max_matches = int(np.floor(N * max_selectivity))

    if min_matches > max_matches:
        raise ValueError(
            f"Invalid selectivity range: "
            f"min_matches={min_matches}, max_matches={max_matches}. "
            f"Try lowering topk or widening the selectivity range."
        )

    print(
        f"[INFO] Target selectivity range: "
        f"{min_selectivity * 100:.2f}% - {max_selectivity * 100:.2f}% "
        f"= [{min_matches}, {max_matches}] matches"
    )

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
                print(
                    f"[WARN] Exhausted all {N} source titles after "
                    f"{written} queries."
                )
                break

            query_vec = vectors[qid]

            rx_str, candidates = generate_one_query(
                title_idx=qid,
                titles=titles,
                style=style,
                min_matches=min_matches,
                max_matches=max_matches,
                max_attempts=max_attempts,
            )

            if rx_str is None or not candidates:
                skipped += 1
                continue

            cand_count = len(candidates)
            selectivity = cand_count / N

            # Ground truth: top-K by exact Euclidean distance among regex matches
            cand_vecs = vectors[candidates]
            diffs = cand_vecs - query_vec
            dists = np.linalg.norm(diffs, axis=1)
            order = np.argsort(dists)[:topk]
            top_k_ids = [candidates[i] for i in order]

            vec_str = " ".join(f"{x:.6f}" for x in query_vec)
            fq.write(f"{rx_str} {vec_str}\n")

            fg.write(" ".join(str(i) for i in top_k_ids) + "\n")

            written += 1
            pbar.update(1)

            print(
                f"[ACCEPT] {written}/{n_queries}: "
                f"regex='{rx_str}', "
                f"matches={cand_count}, "
                f"selectivity={selectivity * 100:.3f}%"
            )

        pbar.close()

    print(f"[INFO] Wrote {written} queries to '{query_out}'")
    print(f"[INFO] Wrote {written} ground-truth lines to '{gt_out}'")
    print(f"[INFO] Skipped {skipped} source titles / attempts")

    if written < n_queries:
        print(
            f"[WARN] Only generated {written}/{n_queries} queries. "
            f"Try increasing --max_attempts or using --style all."
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RegExANN queries and ground-truth with selectivity control"
    )

    parser.add_argument(
        "title_file",
        help="Path to strings .txt file, one string per line"
    )
    parser.add_argument(
        "fvec_file",
        help="Path to .fvecs vector file"
    )
    parser.add_argument(
        "query_out",
        help="Output query file path"
    )
    parser.add_argument(
        "gt_out",
        help="Output ground-truth file path"
    )

    parser.add_argument(
        "--n_queries",
        type=int,
        default=100,
        help="Number of queries to generate"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K ground-truth results per query"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="mixed",
        choices=STYLES + ["all"],
        help="Regex style. Use 'all' to cycle evenly."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=30,
        help="Max title attempts per source query before skipping"
    )
    parser.add_argument(
        "--min_selectivity",
        type=float,
        default=0.01,
        help="Minimum regex selectivity, e.g., 0.01 for 1%"
    )
    parser.add_argument(
        "--max_selectivity",
        type=float,
        default=0.10,
        help="Maximum regex selectivity, e.g., 0.10 for 10%"
    )

    args = parser.parse_args()

    if args.min_selectivity < 0 or args.max_selectivity > 1:
        raise ValueError("Selectivity must be within [0, 1].")

    if args.min_selectivity > args.max_selectivity:
        raise ValueError(
            f"min_selectivity={args.min_selectivity} cannot be larger than "
            f"max_selectivity={args.max_selectivity}."
        )

    if args.style == "all":
        style_cycle = [STYLES[i % len(STYLES)] for i in range(args.n_queries)]
        original_generate_one_query = generate_one_query

        counter = {"idx": 0}

        def cycling_generate_one_query(
            title_idx: int,
            titles: List[str],
            style: str,
            min_matches: int,
            max_matches: int,
            max_attempts: int = 30,
        ) -> Tuple[Optional[str], Optional[List[int]]]:
            s = style_cycle[counter["idx"] % len(style_cycle)]
            counter["idx"] += 1

            return original_generate_one_query(
                title_idx=title_idx,
                titles=titles,
                style=s,
                min_matches=min_matches,
                max_matches=max_matches,
                max_attempts=max_attempts,
            )

        generate_one_query = cycling_generate_one_query  # noqa: F811

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
        min_selectivity=args.min_selectivity,
        max_selectivity=args.max_selectivity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Usage examples
# ─────────────────────────────────────────────────────────────────────────────
#
# Generate mixed queries with selectivity between 1% and 10%:
#
# /usr/bin/python3 scripts/gen_query.py \
#   dataset/gist/strings.txt \
#   dataset/gist/vectors.fvecs \
#   dataset/gist/query_sel1_10.txt \
#   dataset/gist/groundtruth_sel1_10.txt \
#   --n_queries 100 \
#   --topk 10 \
#   --style mixed \
#   --min_selectivity 0.01 \
#   --max_selectivity 0.10 \
#   --max_attempts 200 \
#   --seed 42
#
# Generate all styles evenly:
#
# /usr/bin/python3 scripts/gen_query.py \
#   dataset/gist/strings.txt \
#   dataset/gist/vectors.fvecs \
#   dataset/gist/query_sel1_10.txt \
#   dataset/gist/groundtruth_sel1_10.txt \
#   --n_queries 100 \
#   --topk 10 \
#   --style all \
#   --min_selectivity 0.01 \
#   --max_selectivity 0.10 \
#   --max_attempts 300 \
#   --seed 42
#