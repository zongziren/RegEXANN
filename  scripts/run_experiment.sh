#!/usr/bin/env bash
# scripts/run_experiment.sh
# RegExANN – full experiment pipeline for one dataset.
#
# Usage:
#   bash scripts/run_experiment.sh <dataset_name> [options]
#
# Required environment variables (or pass as args):
#   DATASET   – dataset name (arxiv | sift1m | gist1m | laion1m | words | ...)
#   VEC_FILE  – path to vectors.fvecs
#   STR_FILE  – path to strings.txt
#   QRY_FILE  – path to queries.txt
#   K         – number of nearest neighbours (default 10)
#   CLUSTERS  – k-means clusters (default 100)
#   MAX_ITER  – k-means iterations (default 30)
#   PQ_M      – PQ subspaces (default 8)
#   OUTDIR    – output directory (default results/<dataset>)
#
# What this script does:
#   1. Generate ground truth (full-scan exact search)
#   2. Run RegExANN
#   3. Run pre-filter baseline
#   4. Run post-filter baseline
#   5. Print recall@K for all methods
#
# Example:
#   DATASET=arxiv \
#   VEC_FILE=dataset/arxiv/vectors.fvecs \
#   STR_FILE=dataset/arxiv/strings.txt \
#   QRY_FILE=dataset/arxiv/queries.txt \
#   bash scripts/run_experiment.sh

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
DATASET="${DATASET:-dataset}"
VEC_FILE="${VEC_FILE:?'Set VEC_FILE'}"
STR_FILE="${STR_FILE:?'Set STR_FILE'}"
QRY_FILE="${QRY_FILE:?'Set QRY_FILE'}"
K="${K:-10}"
CLUSTERS="${CLUSTERS:-100}"
MAX_ITER="${MAX_ITER:-30}"
PQ_M="${PQ_M:-8}"
OVERSAMPLE="${OVERSAMPLE:-10}"
OUTDIR="${OUTDIR:-results/${DATASET}}"
BIN="${BIN:-./build/regann}"

mkdir -p "${OUTDIR}"

echo "============================================================"
echo "  RegExANN Experiment: ${DATASET}"
echo "  K=${K}  clusters=${CLUSTERS}  pq_m=${PQ_M}"
echo "  Output: ${OUTDIR}"
echo "============================================================"

GT_FILE="${OUTDIR}/groundtruth.txt"
ANN_FILE="${OUTDIR}/ann.txt"
PRE_FILE="${OUTDIR}/prefilter.txt"
POST_FILE="${OUTDIR}/postfilter.txt"

# ── 1. Ground truth ────────────────────────────────────────────────────────
echo ""
echo "[STEP 1] Generating ground truth …"
"${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
    "${K}" "${CLUSTERS}" "${GT_FILE}" "${MAX_ITER}" \
    groundtruth

# ── 2. RegExANN ────────────────────────────────────────────────────────────
echo ""
echo "[STEP 2] Running RegExANN …"
"${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
    "${K}" "${CLUSTERS}" "${ANN_FILE}" "${MAX_ITER}" \
    ann "pq_m=${PQ_M}" "gt=${GT_FILE}"

# ── 3. Pre-filter baseline ─────────────────────────────────────────────────
echo ""
echo "[STEP 3] Running pre-filter baseline …"
"${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
    "${K}" "${CLUSTERS}" "${PRE_FILE}" "${MAX_ITER}" \
    prefilter "gt=${GT_FILE}"

# ── 4. Post-filter baseline ────────────────────────────────────────────────
echo ""
echo "[STEP 4] Running post-filter baseline (oversample=${OVERSAMPLE}) …"
"${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
    "${K}" "${CLUSTERS}" "${POST_FILE}" "${MAX_ITER}" \
    postfilter "oversample=${OVERSAMPLE}" "gt=${GT_FILE}"

# ── 5. Recall summary ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RECALL@${K} SUMMARY"
echo "============================================================"

for METHOD in ann prefilter postfilter; do
    FILE="${OUTDIR}/${METHOD}.txt"
    if [ -f "${FILE}" ]; then
        echo ""
        echo "  [${METHOD}]"
        python3 tools/eval_recall.py --gt "${GT_FILE}" --pred "${FILE}" --K "${K}"
    fi
done

echo ""
echo "Done. Results written to ${OUTDIR}/"
