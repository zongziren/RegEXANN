#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SIFT regex pattern-class experiment
#
# Question:
#   How do different regex pattern classes affect RegExANN?
#
# This script:
#   1. Generates strict-style queries and groundtruth.
#   2. Loads an existing SIFT index from results/sift/idx/sift.
#   3. Runs RegExANN with multiple ef values.
#   4. Writes all results to one summary CSV.
# ============================================================

BIN=./exp/build/regann
GEN=dataset/gen_query_strict_style.py

DATASET=sift
VEC=dataset/sift/vectors.fvecs
STR=dataset/sift/strings.txt

K=10
CLUSTERS=500
MAX_ITER=30

OUTDIR=results/sift_pattern_classes

# Load the existing index built by the main SIFT experiment.
# This script never saves or overwrites the index.
IDX="results/sift/idx/sift"

mkdir -p "${OUTDIR}/logs" "${OUTDIR}/queries" "${OUTDIR}/gt"

CSV="${OUTDIR}/summary.csv"
echo "pattern_class,method,param,param_value,recall_pct,avg_time_ms,qps,peak_mem_mb,idx_size_mb" > "${CSV}"

PATTERNS=(
    substring
    prefix
    suffix
    alternation
    wildcard
    mixed
)

N_QUERIES=10

MIN_SEL=0.001
MAX_SEL=0.20
MAX_ATTEMPTS=1000
SEED=42

ANN_EFS=(10 20 50 100 200)

append_row() {
    local pattern="$1"
    local method="$2"
    local param="$3"
    local param_val="$4"
    local recall="$5"
    local avg_time="$6"
    local qps="$7"
    local peak_mem_mb="$8"
    local idx_size_mb="$9"

    echo "${pattern},${method},${param},${param_val},${recall},${avg_time},${qps},${peak_mem_mb},${idx_size_mb}" >> "${CSV}"
}

generate_query_and_gt() {
    local pattern="$1"
    local qry="$2"
    local gt="$3"

    echo "  → generating strict ${pattern} queries"

    /usr/bin/python3 "${GEN}" \
        "${STR}" \
        "${VEC}" \
        "${qry}" \
        "${gt}" \
        --n_queries "${N_QUERIES}" \
        --topk "${K}" \
        --style "${pattern}" \
        --min_selectivity "${MIN_SEL}" \
        --max_selectivity "${MAX_SEL}" \
        --max_attempts "${MAX_ATTEMPTS}" \
        --seed "${SEED}"
}

run_ann() {
    local pattern="$1"
    local ef="$2"
    local qry="$3"
    local gt="$4"
    local out="$5"

    local log="${OUTDIR}/logs/${pattern}_ann_ef${ef}.log"
    local timelog="${OUTDIR}/logs/${pattern}_ann_ef${ef}.time.log"

    echo "  → pattern=${pattern}  method=ann  ef=${ef}"

    if [ ! -f "${IDX}.kmidx" ]; then
        echo "[ERROR] Missing existing index: ${IDX}.kmidx"
        echo "[ERROR] This script is load-only and will not build or save an index."
        echo "[ERROR] Please make sure the SIFT index exists under results/sift/idx/."
        exit 1
    fi

    /usr/bin/time -v -o "${timelog}" "${BIN}" "${VEC}" "${STR}" "${qry}" \
        "${K}" "${CLUSTERS}" "${out}" "${MAX_ITER}" \
        ann pq_m=8 "ef=${ef}" "load=${IDX}" "gt=${gt}" \
        2>&1 | tee "${log}"

    local recall avg_time qps peak_mem_kb peak_mem_mb idx_size_mb

    recall=$(grep '\[EVAL\] Recall' "${log}" | grep -oP '[\d.]+(?= %)' | head -1 || true)
    avg_time=$(grep -oP '(?<=Avg total time   : )[\d.]+|(?<=Avg: )[\d.]+' "${log}" | head -1 || true)
    qps=$(grep -oP '(?<=QPS              : )[\d.]+|(?<=QPS: )[\d.]+' "${log}" | head -1 || true)

    peak_mem_kb=$(grep -oP "(?<=Maximum resident set size \(kbytes\): )\d+" "${timelog}" | head -1 || true)

    if [ -n "${peak_mem_kb:-}" ]; then
        peak_mem_mb=$(awk -v kb="${peak_mem_kb}" 'BEGIN { printf "%.2f", kb / 1024 }')
    else
        peak_mem_mb="N/A"
    fi

    if [ -d "$(dirname "${IDX}")" ]; then
        idx_size_mb=$(du -sm "$(dirname "${IDX}")" | awk '{print $1}')
    else
        idx_size_mb="N/A"
    fi

    recall="${recall:-N/A}"
    avg_time="${avg_time:-N/A}"
    qps="${qps:-N/A}"
    peak_mem_mb="${peak_mem_mb:-N/A}"
    idx_size_mb="${idx_size_mb:-N/A}"

    append_row "${pattern}" "ann" "ef" "${ef}" \
        "${recall}" "${avg_time}" "${qps}" "${peak_mem_mb}" "${idx_size_mb}"

    echo "     recall=${recall}%  avg_time=${avg_time}ms  QPS=${qps}  peak_mem=${peak_mem_mb}MB  idx_size=${idx_size_mb}MB"
    echo ""
}

echo "════════════════════════════════════════════════════════════"
echo " SIFT regex pattern-class experiment"
echo " RegExANN only, multiple ef values"
echo " patterns       = ${PATTERNS[*]}"
echo " ef values      = ${ANN_EFS[*]}"
echo " n_queries      = ${N_QUERIES}"
echo " min_selectivity= ${MIN_SEL}"
echo " max_selectivity= ${MAX_SEL}"
echo " index          = ${IDX}"
echo " output         = ${OUTDIR}"
echo "════════════════════════════════════════════════════════════"

if [ ! -f "${IDX}.kmidx" ]; then
    echo "[ERROR] Missing existing index: ${IDX}.kmidx"
    echo "[ERROR] Build the main SIFT index first, or correct IDX in this script."
    exit 1
fi

for P in "${PATTERNS[@]}"; do
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "[1] Generate queries + groundtruth: ${P}"
    echo "────────────────────────────────────────────────────────────"

    QRY="${OUTDIR}/queries/sift_${P}.txt"
    GT="${OUTDIR}/gt/sift_${P}_groundtruth.txt"

    rm -f "${QRY}" "${GT}"
    generate_query_and_gt "${P}" "${QRY}" "${GT}"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "[2] Run RegExANN ef sweep: ${P}"
    echo "────────────────────────────────────────────────────────────"

    for EF in "${ANN_EFS[@]}"; do
        OUT_ANN="${OUTDIR}/sift_${P}_ann_ef${EF}.txt"
        run_ann "${P}" "${EF}" "${QRY}" "${GT}" "${OUT_ANN}"
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo " Summary: SIFT regex pattern classes"
echo "════════════════════════════════════════════════════════════"

printf "  %-13s %-18s %8s %12s %10s %12s %12s\n" \
    "Pattern" "Method" "Recall%" "Avg(ms)" "QPS" "Mem(MB)" "Idx(MB)"

printf "  %-13s %-18s %8s %12s %10s %12s %12s\n" \
    "────────────" "──────────────────" "────────" "────────────" "──────────" "────────────" "────────────"

while IFS=, read -r pattern method param pval recall avg_time qps peak_mem_mb idx_size_mb; do
    [ "${pattern}" = "pattern_class" ] && continue

    printf "  %-13s %-18s %8s %12s %10s %12s %12s\n" \
        "${pattern}" "${method} ${param}=${pval}" "${recall}" "${avg_time}" "${qps}" "${peak_mem_mb}" "${idx_size_mb}"
done < "${CSV}"

echo ""
echo "CSV         → ${CSV}"
echo "Logs        → ${OUTDIR}/logs/"
echo "Query files → ${OUTDIR}/queries/"
echo "GT files    → ${OUTDIR}/gt/"
