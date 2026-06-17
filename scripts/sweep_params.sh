#!/usr/bin/env bash
# scripts/sweep_params.sh
# ─────────────────────────────────────────────────────────────────────────────
# Sweep the three new recall/speed parameters across all algorithms:
#   ann        : ef values
#   prefilter  : sample_ratio values
#   postfilter : oversample values (with adaptive expansion)
#
# Usage:
#   VEC_FILE=dataset/arxiv/vectors.fvecs \
#   STR_FILE=dataset/arxiv/strings.txt \
#   QRY_FILE=dataset/arxiv/queries.txt \
#   GT_FILE=results/arxiv/gt.txt \
#   bash scripts/sweep_params.sh
#
# Output: results/<dataset>/sweep_params.csv
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VEC_FILE="${VEC_FILE:?'Set VEC_FILE'}"
STR_FILE="${STR_FILE:?'Set STR_FILE'}"
QRY_FILE="${QRY_FILE:?'Set QRY_FILE'}"
GT_FILE="${GT_FILE:?'Set GT_FILE'}"
K="${K:-10}"
CLUSTERS="${CLUSTERS:-100}"
MAX_ITER="${MAX_ITER:-30}"
PQ_M="${PQ_M:-8}"
DATASET="${DATASET:-dataset}"
OUTDIR="${OUTDIR:-results/${DATASET}}"
BIN="${BIN:-./build/regann}"
EVAL="${EVAL:-./build/eval_recall}"

# Sweep ranges
EF_LIST="${EF_LIST:-${K} 20 50 100 200}"
SAMPLE_RATIOS="${SAMPLE_RATIOS:-1.0 0.5 0.2 0.1}"
OVERSAMPLE_LIST="${OVERSAMPLE_LIST:-5 10 20 50}"
MAX_EXPANSION="${MAX_EXPANSION:-0}"   # 0 = unlimited adaptive expansion

mkdir -p "${OUTDIR}/sweep_params"

CSV="${OUTDIR}/sweep_params.csv"
echo "algorithm,param,param_value,recall,avg_time_ms" | tee "${CSV}"

# ── Helper: run binary, capture recall + avg_time ─────────────────────────────
run_and_record() {
    local algo="$1" param_name="$2" param_val="$3"
    shift 3
    local extra_args=("$@")
    local out="${OUTDIR}/sweep_params/${algo}_${param_name}${param_val}.txt"
    param_val_safe=$(echo "${param_val}" | tr '.' 'p')
    out="${OUTDIR}/sweep_params/${algo}_${param_name}${param_val_safe}.txt"

    LOG=$("${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
        "${K}" "${CLUSTERS}" "${out}" "${MAX_ITER}" \
        "${algo}" "${extra_args[@]}" "gt=${GT_FILE}" 2>&1 | tee /dev/stderr)

    AVG_TIME=$(echo "${LOG}" | grep "Avg total time" | awk '{print $NF}' | tr -d 'ms') || AVG_TIME=""
    RECALL=$("${EVAL}" --gt "${GT_FILE}" --pred "${out}" --K "${K}" 2>/dev/null \
        | grep "Mean" | awk '{print $NF}' | tr -d '%') || RECALL=""

    echo "${algo},${param_name},${param_val},${RECALL},${AVG_TIME}" | tee -a "${CSV}"
}

# ── Build index once (reuse for all ef values) ────────────────────────────────
echo ""
echo "══ Building RegExANN index (k=${CLUSTERS}, pq_m=${PQ_M}) ══"
IDX_PREFIX="${OUTDIR}/sweep_params/idx"
"${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
    "${K}" "${CLUSTERS}" "${OUTDIR}/sweep_params/_build_dummy.txt" "${MAX_ITER}" \
    ann "pq_m=${PQ_M}" "ef=${K}" "save=${IDX_PREFIX}" "gt=${GT_FILE}" 2>&1

# ── Sweep ef (ann) ────────────────────────────────────────────────────────────
echo ""
echo "══ Sweeping ef for ann: ${EF_LIST} ══"
for EF in ${EF_LIST}; do
    out="${OUTDIR}/sweep_params/ann_ef${EF}.txt"
    LOG=$("${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
        "${K}" "${CLUSTERS}" "${out}" "${MAX_ITER}" \
        ann "pq_m=${PQ_M}" "ef=${EF}" "load=${IDX_PREFIX}" "gt=${GT_FILE}" \
        2>&1 | tee /dev/stderr)
    AVG_TIME=$(echo "${LOG}" | grep "Avg total time" | awk '{print $NF}' | tr -d 'ms') || AVG_TIME=""
    RECALL=$("${EVAL}" --gt "${GT_FILE}" --pred "${out}" --K "${K}" 2>/dev/null \
        | grep "Mean" | awk '{print $NF}' | tr -d '%') || RECALL=""
    echo "ann,ef,${EF},${RECALL},${AVG_TIME}" | tee -a "${CSV}"
done

# ── Sweep sample_ratio (prefilter) ────────────────────────────────────────────
echo ""
echo "══ Sweeping sample_ratio for prefilter: ${SAMPLE_RATIOS} ══"
for SR in ${SAMPLE_RATIOS}; do
    SR_TAG=$(echo "${SR}" | tr '.' 'p')
    out="${OUTDIR}/sweep_params/prefilter_sr${SR_TAG}.txt"
    LOG=$("${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
        "${K}" "${CLUSTERS}" "${out}" "${MAX_ITER}" \
        prefilter "sample_ratio=${SR}" "gt=${GT_FILE}" \
        2>&1 | tee /dev/stderr)
    AVG_TIME=$(echo "${LOG}" | grep "Avg total time\|Avg:" | awk '{print $NF}' | tr -d 'ms' | head -1) || AVG_TIME=""
    RECALL=$("${EVAL}" --gt "${GT_FILE}" --pred "${out}" --K "${K}" 2>/dev/null \
        | grep "Mean" | awk '{print $NF}' | tr -d '%') || RECALL=""
    echo "prefilter,sample_ratio,${SR},${RECALL},${AVG_TIME}" | tee -a "${CSV}"
done

# ── Sweep oversample (postfilter, adaptive) ───────────────────────────────────
echo ""
echo "══ Sweeping oversample for postfilter (max_expansion=${MAX_EXPANSION}): ${OVERSAMPLE_LIST} ══"
for OV in ${OVERSAMPLE_LIST}; do
    out="${OUTDIR}/sweep_params/postfilter_ov${OV}.txt"
    LOG=$("${BIN}" "${VEC_FILE}" "${STR_FILE}" "${QRY_FILE}" \
        "${K}" "${CLUSTERS}" "${out}" "${MAX_ITER}" \
        postfilter "oversample=${OV}" "max_expansion=${MAX_EXPANSION}" "gt=${GT_FILE}" \
        2>&1 | tee /dev/stderr)
    AVG_TIME=$(echo "${LOG}" | grep "Avg total time\|Avg:" | awk '{print $NF}' | tr -d 'ms' | head -1) || AVG_TIME=""
    RECALL=$("${EVAL}" --gt "${GT_FILE}" --pred "${out}" --K "${K}" 2>/dev/null \
        | grep "Mean" | awk '{print $NF}' | tr -d '%') || RECALL=""
    echo "postfilter,oversample,${OV},${RECALL},${AVG_TIME}" | tee -a "${CSV}"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Parameter sweep complete."
echo "  Results: ${CSV}"
echo "════════════════════════════════════════════════════════════════"
