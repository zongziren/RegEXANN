#!/usr/bin/env bash
# scripts/run_experiment.sh
# RegExANN — full experiment pipeline for one dataset.
#
# Usage:
#   DATASET=arxiv \
#   VEC_FILE=dataset/arxiv/vectors.fvecs \
#   STR_FILE=dataset/arxiv/strings.txt \
#   QRY_FILE=dataset/arxiv/queries.txt \
#   bash scripts/run_experiment.sh
#
# Environment variables:
#   DATASET          dataset name (arxiv | sift1m | gist1m | ...)
#   VEC_FILE         path to vectors.fvecs
#   STR_FILE         path to strings.txt
#   QRY_FILE         path to queries.txt
#   K                number of nearest neighbours (default 10)
#   CLUSTERS         k-means clusters (default 100)
#   MAX_ITER         k-means iterations (default 30)
#   PQ_M             PQ subspaces (default 8)
#   EF               ann candidate pool size — default K (original behaviour)
#   OVERSAMPLE       postfilter initial oversample factor (default 10)
#   MAX_EXPANSION    postfilter max expansion cap, 0=unlimited (default 0)
#   SAMPLE_RATIO     prefilter match-set fraction, 1.0=exact (default 1.0)
#   OUTDIR           output directory (default results/<dataset>)
#
# What this script does:
#   1. Generate ground truth (full-scan exact search)
#   2. Run RegExANN  (with ef parameter)
#   3. Run pre-filter baseline  (with sample_ratio)
#   4. Run post-filter baseline  (with adaptive expansion)
#   5. Print Recall@K summary for all methods

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET="${DATASET:-dataset}"
VEC_FILE="${VEC_FILE:?'Set VEC_FILE'}"
STR_FILE="${STR_FILE:?'Set STR_FILE'}"
QRY_FILE="${QRY_FILE:?'Set QRY_FILE'}"
K="${K:-10}"
CLUSTERS="${CLUSTERS:-100}"
MAX_ITER="${MAX_ITER:-30}"
PQ_M="${PQ_M:-8}"
EF="${EF:-${K}}"             # ann candidate pool (ef=K → original behaviour)
OVERSAMPLE="${OVERSAMPLE:-10}"
MAX_EXPANSION="${MAX_EXPANSION:-0}"   # 0 = unlimited adaptive expansion
SAMPLE_RATIO="${SAMPLE_RATIO:-1.0}"  # 1.0 = 100% recall (original behaviour)
OUTDIR="${OUTDIR:-results/${DATASET}}"
BIN="${BIN:-./build/regann}"

mkdir -p "${OUTDIR}"

echo "============================================================"
echo "  RegExANN Experiment: ${DATASET}"
echo "  K=${K}  clusters=${CLUSTERS}  pq_m=${PQ_M}"
echo "  ann ef=${EF}"
echo "  prefilter sample_ratio=${SAMPLE_RATIO}"
echo "  postfilter oversample=${OVERSAMPLE}  max_expansion=${MAX_EXPANSION}"
echo "  Output: ${OUTDIR}"
echo "============================================================"

GT_FILE="${OUTDIR}/groundtruth.txt"
ANN_FILE="${OUTDIR}/ann.txt"
PRE_FILE="${OUTDIR}/prefilter.txt"
POST_FILE="${OUTDIR}/postfilter.txt"

# ── 1. Ground truth ───────────────────────────────────────────────────────────
echo ""
echo "[STEP 1] Generating ground truth …"
"${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
    "${K}" "${CLUSTERS}" "${GT_FILE}" "${MAX_ITER}" \
    groundtruth

# ── 2. RegExANN ───────────────────────────────────────────────────────────────
echo ""
echo "[STEP 2] Running RegExANN (ef=${EF}) …"
"${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
    "${K}" "${CLUSTERS}" "${ANN_FILE}" "${MAX_ITER}" \
    ann \
    "pq_m=${PQ_M}" \
    "ef=${EF}" \
    "gt=${GT_FILE}"

# ── 3. Pre-filter baseline ────────────────────────────────────────────────────
echo ""
echo "[STEP 3] Running pre-filter baseline (sample_ratio=${SAMPLE_RATIO}) …"
"${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
    "${K}" "${CLUSTERS}" "${PRE_FILE}" "${MAX_ITER}" \
    prefilter \
    "sample_ratio=${SAMPLE_RATIO}" \
    "gt=${GT_FILE}"

# ── 4. Post-filter baseline ───────────────────────────────────────────────────
echo ""
echo "[STEP 4] Running post-filter baseline (oversample=${OVERSAMPLE}, max_expansion=${MAX_EXPANSION}) …"
"${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
    "${K}" "${CLUSTERS}" "${POST_FILE}" "${MAX_ITER}" \
    postfilter \
    "oversample=${OVERSAMPLE}" \
    "max_expansion=${MAX_EXPANSION}" \
    "gt=${GT_FILE}"

# ── 5. Recall summary ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RECALL@${K} SUMMARY  [${DATASET}]"
echo "============================================================"
printf "  %-38s  %s\n" "Method" "Recall@${K}"
printf "  %-38s  %s\n" "──────────────────────────────────────" "────────────"

for METHOD_LABEL in \
    "${ANN_FILE}|RegExANN (ef=${EF})" \
    "${PRE_FILE}|Pre-filter (sample_ratio=${SAMPLE_RATIO})" \
    "${POST_FILE}|Post-filter (oversample=${OVERSAMPLE}, max_exp=${MAX_EXPANSION})"
do
    FILE="${METHOD_LABEL%%|*}"
    LABEL="${METHOD_LABEL##*|}"
    if [ -f "${FILE}" ]; then
        R=$(python3 tools/eval_recall.py \
            --gt "${GT_FILE}" --pred "${FILE}" --K "${K}" 2>/dev/null \
            | grep -i "mean" | awk '{print $NF}') || R="(error)"
        printf "  %-38s  %s\n" "${LABEL}" "${R}"
    else
        printf "  %-38s  (not found)\n" "${LABEL}"
    fi
done

echo ""
echo "Done. Results written to ${OUTDIR}/"
