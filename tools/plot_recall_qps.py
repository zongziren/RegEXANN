#!/usr/bin/env python3
"""tools/plot_recall_qps.py
Reproduce the Recall@K vs QPS trade-off plots from the paper (Figure 5).

For each method, the script reads the timing logs from a results directory
(produced by run_experiment.sh) and plots a Recall-QPS curve by varying the
cluster count (for RegExANN) or oversample factor (for post-filter).

Usage
-----
python3 tools/plot_recall_qps.py \\
    --results_dir results/arxiv \\
    --K 10 \\
    --dataset arXiv \\
    --output plots/arxiv_recall_qps.png

The results directory should contain:
    groundtruth.txt
    ann_c<N>.txt   (for various cluster counts N, produced by sweep_clusters.sh)
    prefilter.txt
    postfilter_ov<M>.txt  (for various oversample factors M)
    sweep_clusters.csv    (produced by sweep_clusters.sh)

Alternatively pass a simple summary CSV with columns:
    method,recall,qps
"""

import argparse
import os
import sys

def load_csv(path):
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: v for k, v in row.items()})
    return rows


def main():
    p = argparse.ArgumentParser(description='Plot Recall@K vs QPS.')
    p.add_argument('--results_dir', required=True)
    p.add_argument('--K',           type=int, default=10)
    p.add_argument('--dataset',     default='Dataset')
    p.add_argument('--output',      default=None,
                   help='Output image path (default: show interactively)')
    args = p.parse_args()

    try:
        import matplotlib
        if args.output:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        sys.exit('[ERROR] Install matplotlib: pip install matplotlib')

    rdir = args.results_dir
    fig, ax = plt.subplots(figsize=(6, 4))

    # ── RegExANN curve (varying clusters) ────────────────────────────────
    sweep_csv = os.path.join(rdir, 'sweep_clusters.csv')
    if os.path.exists(sweep_csv):
        rows = load_csv(sweep_csv)
        recalls, qpss = [], []
        for r in sorted(rows, key=lambda x: float(x['recall'])):
            try:
                recall = float(r['recall']) / 100.0
                t_ms   = float(r['avg_time_ms'])
                qps    = 1000.0 / t_ms if t_ms > 0 else 0
                recalls.append(recall)
                qpss.append(qps)
            except (ValueError, KeyError):
                continue
        if recalls:
            ax.plot(recalls, qpss, 'o-', label='REGEXANN', color='tab:blue', lw=2)

    # ── Pre-filter (single point) ─────────────────────────────────────────
    summary_csv = os.path.join(rdir, 'summary.csv')
    if os.path.exists(summary_csv):
        for row in load_csv(summary_csv):
            try:
                rc  = float(row['recall']) / 100.0
                qps = float(row['qps'])
                m   = row.get('method', '?')
                marker = {'prefilter': 's', 'postfilter': '^',
                          'fullscan': 'D'}.get(m, 'x')
                color  = {'prefilter': 'tab:orange', 'postfilter': 'tab:green',
                          'fullscan': 'tab:red'}.get(m, 'gray')
                label  = {'prefilter': 'Pre-filter', 'postfilter': 'Post-filter',
                          'fullscan': 'Full Scan'}.get(m, m)
                ax.scatter([rc], [qps], marker=marker, color=color,
                           s=80, label=label, zorder=5)
            except (ValueError, KeyError):
                continue

    ax.set_xlabel(f'Recall@{args.K}', fontsize=12)
    ax.set_ylabel('QPS (queries/sec)', fontsize=12)
    ax.set_title(f'{args.dataset}', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        plt.savefig(args.output, dpi=150)
        print(f'[plot] Saved → {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
