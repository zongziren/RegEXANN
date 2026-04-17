#!/usr/bin/env python3
"""tools/eval_recall.py
Compute Recall@K comparing a system's output against ground truth.

Usage
-----
python3 tools/eval_recall.py \\
    --gt    results/gt.txt     \\
    --pred  results/ann_out.txt \\
    --K     10                 \\
    [--verbose]

Output
------
Prints mean Recall@K and optionally per-query recall.
Exits with code 0 on success, 1 on error.
"""

import argparse
import sys


def load_result_file(path: str):
    results = []
    with open(path) as f:
        for line in f:
            ids = [int(x) for x in line.split() if x.strip()]
            results.append(ids)
    return results


def recall_at_k(gt_ids, pred_ids, K):
    """Recall@K for a single query."""
    if not gt_ids:
        return 1.0          # query matched nothing → trivially correct
    true_set = set(gt_ids[:K])
    hits = sum(1 for x in pred_ids if x in true_set)
    return hits / len(true_set)


def main():
    p = argparse.ArgumentParser(description='Compute Recall@K.')
    p.add_argument('--gt',      required=True, help='Ground-truth result file')
    p.add_argument('--pred',    required=True, help='Predicted result file')
    p.add_argument('--K',       type=int, default=10)
    p.add_argument('--verbose', action='store_true',
                   help='Print per-query recall')
    args = p.parse_args()

    gt   = load_result_file(args.gt)
    pred = load_result_file(args.pred)

    if len(gt) != len(pred):
        print(f'[ERROR] Query count mismatch: gt={len(gt)}, pred={len(pred)}',
              file=sys.stderr)
        sys.exit(1)

    per_query = []
    for i, (g, r) in enumerate(zip(gt, pred)):
        rc = recall_at_k(g, r, args.K)
        per_query.append(rc)
        if args.verbose:
            print(f'Q{i:04d}  recall@{args.K} = {rc:.4f}'
                  f'  (gt={len(g[:args.K])} / pred={len(r)})')

    mean = sum(per_query) / len(per_query) if per_query else 0.0
    perfect = sum(1 for r in per_query if r >= 1.0 - 1e-9)

    print(f'\n=== Recall@{args.K} Report ===')
    print(f'  Queries      : {len(per_query)}')
    print(f'  Mean recall  : {mean*100:.2f} %')
    print(f'  Perfect (1.0): {perfect} / {len(per_query)}'
          f' ({100*perfect/max(len(per_query),1):.1f} %)')
    print(f'  Min recall   : {min(per_query)*100:.2f} %')
    print(f'  Max recall   : {max(per_query)*100:.2f} %')


if __name__ == '__main__':
    main()
