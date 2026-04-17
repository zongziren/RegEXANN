#!/usr/bin/env python3
"""tools/gen_queries.py
Generate query files for RegExANN experiments.

Each output line: <regex> <v1> <v2> … <vd>

The script supports several regex pattern styles:
  substring   – literal substring match  (e.g. "neural")
  prefix      – prefix match             (e.g. "^deep")
  suffix      – suffix match             (e.g. "search$")
  alternation – OR of two words          (e.g. "graph|network")
  wildcard    – word.*word pattern       (e.g. "deep.*learning")
  mixed       – alternation of wildcards (e.g. "ann.*gpu|gpu.*ann")

Usage
-----
python3 tools/gen_queries.py \\
    --fvecs  dataset/arxiv/vectors.fvecs \\
    --strings dataset/arxiv/strings.txt  \\
    --output  dataset/arxiv/queries.txt  \\
    --num_queries 1000 \\
    --style substring  \\
    --seed 42

For query vectors the script either:
  (a) samples existing dataset vectors (with optional Gaussian noise), or
  (b) generates random unit vectors of the correct dimension.
"""

import argparse
import random
import struct
import math
import os
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# fvecs I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_fvecs(path: str):
    """Return list of float lists."""
    vecs = []
    with open(path, 'rb') as f:
        while True:
            header = f.read(4)
            if not header:
                break
            dim = struct.unpack('i', header)[0]
            data = f.read(dim * 4)
            if len(data) < dim * 4:
                break
            vecs.append(list(struct.unpack(f'{dim}f', data)))
    return vecs


def load_strings(path: str):
    with open(path, encoding='utf-8', errors='replace') as f:
        return [line.rstrip('\n') for line in f]


# ─────────────────────────────────────────────────────────────────────────────
# Trigram-based word extraction
# ─────────────────────────────────────────────────────────────────────────────

def alpha_words(text: str):
    """Return lower-case alphabetic tokens of length >= 3."""
    import re
    return [w for w in re.split(r'[^a-zA-Z]+', text.lower()) if len(w) >= 3]


def collect_word_pool(strings, min_count: int = 2):
    """Return words that appear in at least `min_count` strings."""
    from collections import Counter
    cnt: Counter = Counter()
    for s in strings:
        for w in set(alpha_words(s)):
            cnt[w] += 1
    return [w for w, c in cnt.items() if c >= min_count]


# ─────────────────────────────────────────────────────────────────────────────
# Query vector generation
# ─────────────────────────────────────────────────────────────────────────────

def noisy_vec(vec, noise_std: float = 0.1):
    return [x + random.gauss(0, noise_std) for x in vec]


def random_unit_vec(dim: int):
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x / norm for x in v]


# ─────────────────────────────────────────────────────────────────────────────
# Regex pattern builders
# ─────────────────────────────────────────────────────────────────────────────

def make_substring(words):
    return random.choice(words)


def make_prefix(words):
    return random.choice(words)   # regex_search handles prefix naturally


def make_suffix(words):
    return random.choice(words) + '$'


def make_alternation(words):
    a, b = random.sample(words, 2)
    return f'{a}|{b}'


def make_wildcard(words):
    a, b = random.sample(words, 2)
    return f'{a}.*{b}'


def make_mixed(words):
    a, b = random.sample(words, 2)
    return f'{a}.*{b}|{b}.*{a}'


BUILDERS = {
    'substring':   make_substring,
    'prefix':      make_prefix,
    'suffix':      make_suffix,
    'alternation': make_alternation,
    'wildcard':    make_wildcard,
    'mixed':       make_mixed,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Generate RegExANN query files.')
    p.add_argument('--fvecs',        required=True,  help='Dataset fvecs file')
    p.add_argument('--strings',      required=True,  help='Dataset strings file')
    p.add_argument('--output',       required=True,  help='Output query file')
    p.add_argument('--num_queries',  type=int, default=100)
    p.add_argument('--style',        default='substring',
                   choices=list(BUILDERS.keys()),
                   help='Regex pattern style')
    p.add_argument('--noise',        type=float, default=0.05,
                   help='Gaussian noise added to sampled query vectors')
    p.add_argument('--random_vecs',  action='store_true',
                   help='Use random unit vectors instead of dataset samples')
    p.add_argument('--min_word_count', type=int, default=2,
                   help='Minimum string appearances for a word to be included')
    p.add_argument('--seed',         type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    print(f'[gen_queries] Loading vectors from {args.fvecs} …')
    vecs = load_fvecs(args.fvecs)
    print(f'[gen_queries] Loading strings from {args.strings} …')
    strs = load_strings(args.strings)

    if not vecs:
        sys.exit('[ERROR] No vectors loaded.')
    dim = len(vecs[0])
    print(f'[gen_queries] {len(vecs)} vectors (dim={dim}), {len(strs)} strings.')

    words = collect_word_pool(strs, min_count=args.min_word_count)
    if len(words) < 2:
        sys.exit('[ERROR] Not enough distinct words in the string corpus.')
    print(f'[gen_queries] Word pool size: {len(words)}')

    builder = BUILDERS[args.style]
    os.makedirs(Path(args.output).parent, exist_ok=True)

    written = 0
    with open(args.output, 'w', encoding='utf-8') as f:
        for _ in range(args.num_queries):
            regex = builder(words)
            if args.random_vecs:
                qv = random_unit_vec(dim)
            else:
                qv = noisy_vec(random.choice(vecs), noise_std=args.noise)
            line = regex + ' ' + ' '.join(f'{x:.8f}' for x in qv)
            f.write(line + '\n')
            written += 1

    print(f'[gen_queries] Wrote {written} queries → {args.output}')


if __name__ == '__main__':
    main()
