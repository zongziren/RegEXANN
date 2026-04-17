#!/usr/bin/env python3
"""tools/prepare_dataset.py
Convert common dataset formats into the fvecs + strings.txt format
expected by RegExANN.

Supported sources
-----------------
  --mode arxiv      arXiv title embeddings  (HDF5 or numpy)
  --mode sift       SIFT fvecs + DBLP strings (random assignment)
  --mode csv        Generic CSV: one row per item, embedding cols + text col
  --mode hdf5       Generic HDF5 with 'train' vectors and 'metadata' strings

Usage examples
--------------
  # arXiv (embeddings in HDF5, titles in txt)
  python3 tools/prepare_dataset.py \\
      --mode arxiv \\
      --emb_file arxiv_embeddings.h5 \\
      --str_file arxiv_titles.txt    \\
      --out_vecs dataset/arxiv/vectors.fvecs \\
      --out_strs dataset/arxiv/strings.txt   \\
      --limit 132687

  # SIFT + random DBLP strings
  python3 tools/prepare_dataset.py \\
      --mode sift \\
      --emb_file dataset/sift/sift_base.fvecs \\
      --str_file dataset/dblp/titles.txt       \\
      --out_vecs dataset/sift1m/vectors.fvecs  \\
      --out_strs dataset/sift1m/strings.txt

  # Generic HDF5
  python3 tools/prepare_dataset.py \\
      --mode hdf5 \\
      --emb_file laion1m.h5 \\
      --vec_key  emb \\
      --str_key  caption \\
      --out_vecs dataset/laion1m/vectors.fvecs \\
      --out_strs dataset/laion1m/strings.txt
"""

import argparse
import os
import random
import struct
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# fvecs I/O
# ─────────────────────────────────────────────────────────────────────────────

def write_fvecs(path: str, vectors):
    """Write an iterable of float lists to fvecs binary format."""
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, 'wb') as f:
        for v in vectors:
            dim = len(v)
            f.write(struct.pack('i', dim))
            f.write(struct.pack(f'{dim}f', *v))


def read_fvecs(path: str):
    vecs = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(4)
            if not h:
                break
            dim = struct.unpack('i', h)[0]
            data = f.read(dim * 4)
            vecs.append(list(struct.unpack(f'{dim}f', data)))
    return vecs


def write_strings(path: str, strings):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for s in strings:
            f.write(s.replace('\n', ' ') + '\n')


def load_strings(path: str):
    with open(path, encoding='utf-8', errors='replace') as f:
        return [line.rstrip('\n') for line in f]


# ─────────────────────────────────────────────────────────────────────────────
# Mode: arxiv
# ─────────────────────────────────────────────────────────────────────────────

def mode_arxiv(args):
    try:
        import h5py
        import numpy as np
    except ImportError:
        sys.exit('[ERROR] Install h5py and numpy: pip install h5py numpy')

    print(f'[prepare] Loading arXiv embeddings from {args.emb_file} …')
    with h5py.File(args.emb_file, 'r') as h:
        key = args.vec_key or 'train'
        vecs = h[key][:]
    if args.limit:
        vecs = vecs[:args.limit]
    print(f'[prepare] Loaded {len(vecs)} vectors (dim={vecs.shape[1]}).')

    print(f'[prepare] Loading titles from {args.str_file} …')
    strings = load_strings(args.str_file)
    if args.limit:
        strings = strings[:args.limit]
    if len(strings) != len(vecs):
        print(f'[WARN] len mismatch: {len(vecs)} vecs, {len(strings)} strings.')
        n = min(len(vecs), len(strings))
        vecs, strings = vecs[:n], strings[:n]

    write_fvecs(args.out_vecs, vecs.tolist())
    write_strings(args.out_strs, strings)
    print(f'[prepare] Wrote {len(vecs)} items → {args.out_vecs}, {args.out_strs}')


# ─────────────────────────────────────────────────────────────────────────────
# Mode: sift (fvecs + random string assignment)
# ─────────────────────────────────────────────────────────────────────────────

def mode_sift(args):
    print(f'[prepare] Reading fvecs from {args.emb_file} …')
    vecs = read_fvecs(args.emb_file)
    if args.limit:
        vecs = vecs[:args.limit]
    print(f'[prepare] Loaded {len(vecs)} vectors.')

    print(f'[prepare] Loading string pool from {args.str_file} …')
    pool = load_strings(args.str_file)
    random.seed(args.seed)
    strings = [random.choice(pool) for _ in range(len(vecs))]

    write_fvecs(args.out_vecs, vecs)
    write_strings(args.out_strs, strings)
    print(f'[prepare] Wrote {len(vecs)} items (random strings).')


# ─────────────────────────────────────────────────────────────────────────────
# Mode: hdf5 (generic)
# ─────────────────────────────────────────────────────────────────────────────

def mode_hdf5(args):
    try:
        import h5py
    except ImportError:
        sys.exit('[ERROR] Install h5py: pip install h5py')

    vec_key = args.vec_key or 'train'
    str_key = args.str_key or 'metadata'

    print(f'[prepare] Opening {args.emb_file} …')
    with h5py.File(args.emb_file, 'r') as h:
        print(f'[prepare] Keys: {list(h.keys())}')
        vecs = h[vec_key][:]
        raw_strs = h[str_key][:]

    strings = [s.decode('utf-8', errors='replace') if isinstance(s, bytes) else str(s)
               for s in raw_strs]

    if args.limit:
        vecs    = vecs[:args.limit]
        strings = strings[:args.limit]

    write_fvecs(args.out_vecs, vecs.tolist())
    write_strings(args.out_strs, strings)
    print(f'[prepare] Wrote {len(vecs)} items → {args.out_vecs}')


# ─────────────────────────────────────────────────────────────────────────────
# Mode: csv
# ─────────────────────────────────────────────────────────────────────────────

def mode_csv(args):
    import csv

    print(f'[prepare] Reading CSV from {args.emb_file} …')
    vecs, strings = [], []
    with open(args.emb_file, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        str_col = args.str_key or 'text'
        for row in reader:
            text = row.pop(str_col, '')
            try:
                vec = [float(v) for v in row.values()]
            except ValueError:
                continue
            vecs.append(vec)
            strings.append(text)
            if args.limit and len(vecs) >= args.limit:
                break

    write_fvecs(args.out_vecs, vecs)
    write_strings(args.out_strs, strings)
    print(f'[prepare] Wrote {len(vecs)} items.')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Prepare RegExANN dataset.')
    p.add_argument('--mode',     required=True,
                   choices=['arxiv', 'sift', 'hdf5', 'csv'])
    p.add_argument('--emb_file', required=True,
                   help='Embedding file (fvecs/HDF5/CSV)')
    p.add_argument('--str_file', default=None,
                   help='String file (for arxiv/sift modes)')
    p.add_argument('--out_vecs', required=True, help='Output fvecs path')
    p.add_argument('--out_strs', required=True, help='Output strings.txt path')
    p.add_argument('--vec_key',  default=None, help='HDF5 key for vectors')
    p.add_argument('--str_key',  default=None, help='HDF5/CSV key for strings')
    p.add_argument('--limit',    type=int, default=None,
                   help='Maximum number of items to load')
    p.add_argument('--seed',     type=int, default=42)
    args = p.parse_args()

    handlers = {
        'arxiv': mode_arxiv,
        'sift':  mode_sift,
        'hdf5':  mode_hdf5,
        'csv':   mode_csv,
    }
    handlers[args.mode](args)


if __name__ == '__main__':
    main()
