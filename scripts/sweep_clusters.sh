#!/usr/bin/env bash
# scripts/sweep_clusters.sh
# Ablation study: vary the number of clusters and measure QPS + Recall@K.
#
# Usage:
#   VEC_FILE=... STR_FILE=... QRY_FILE=... GT_FILE=... \
#   bash scripts/sweep_clusters.sh
#
# Output: CSV to stdout and results/<dataset>/sweep_clusters.csv

set -euo pipefail

VEC_FILE="${VEC_FILE:?}"
STR_FILE="${STR_FILE:?}"
QRY_FILE="${QRY_FILE:?}"
GT_FILE="${GT_FILE:?}"
K="${K:-10}"
MAX_ITER="${MAX_ITER:-30}"
PQ_M="${PQ_M:-8}"
DATASET="${DATASET:-dataset}"
OUTDIR="${OUTDIR:-results/${DATASET}}"
BIN="${BIN:-./build/regann}"

CLUSTER_COUNTS="${CLUSTER_COUNTS:-10 25 50 100 200 400 800}"

mkdir -p "${OUTDIR}/sweep"

CSV="${OUTDIR}/sweep_clusters.csv"
echo "clusters,recall,avg_time_ms" | tee "${CSV}"

for C in ${CLUSTER_COUNTS}; do
    OUT="${OUTDIR}/sweep/ann_c${C}.txt"
    # Capture timing from binary output
    LOG=$("${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
        "${K}" "${C}" "${OUT}" "${MAX_ITER}" \
        ann "pq_m=${PQ_M}" "gt=${GT_FILE}" 2>&1 | tee /dev/stderr)

    AVG_TIME=$(echo "${LOG}" | grep "Avg total time" | awk '{print $NF}' | tr -d 'ms')
    RECALL=$(python3 tools/eval_recall.py \
        --gt "${GT_FILE}" --pred "${OUT}" --K "${K}" 2>/dev/null \
        | grep "Mean recall" | awk '{print $NF}' | tr -d '%')

    echo "${C},${RECALL},${AVG_TIME}" | tee -a "${CSV}"
done

echo ""
echo "Sweep complete. Results: ${CSV}"
