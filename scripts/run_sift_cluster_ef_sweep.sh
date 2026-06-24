#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SIFT cluster-number x ef sweep
#
# This script uses the original SIFT query and groundtruth files:
#   dataset/sift/query.txt
#   dataset/sift/groundtruth.txt
#
# It does not generate new queries.
# ============================================================

BIN=./exp/build/regann

DATASET=sift
VEC=dataset/sift/vectors.fvecs
STR=dataset/sift/strings.txt
QRY=dataset/sift/query.txt
GT=dataset/sift/groundtruth.txt

K=10
MAX_ITER=30
PQ_M=8

OUTDIR=results/sift_cluster_ef_sweep
IDX_DIR="${OUTDIR}/idx"

mkdir -p "${IDX_DIR}" "${OUTDIR}/logs"

CSV="${OUTDIR}/cluster_sift.csv"
echo "dataset,method,param,param_value,ef,recall_pct,avg_time_ms,qps,peak_mem_mb,idx_size_mb" > "${CSV}"

CLUSTER_LIST=(
    10
    20
    50
    100
    200
    500
    1000
    2000
)

EF_LIST=(
    10
    20
    50
    100
    200
    400
    800
    1000
)

check_inputs() {
    if [ ! -x "${BIN}" ]; then
        echo "[ERROR] Missing executable: ${BIN}"
        exit 1
    fi

    for f in "${VEC}" "${STR}" "${QRY}" "${GT}"; do
        if [ ! -f "${f}" ]; then
            echo "[ERROR] Missing file: ${f}"
            exit 1
        fi
    done
}

extract_metrics() {
    local log="$1"
    local timelog="$2"

    local recall avg_time qps peak_mem_kb peak_mem_mb

    recall=$(grep '\[EVAL\] Recall' "${log}" | grep -oP '[\d.]+(?= %)' | head -1 || true)
    avg_time=$(grep -oP '(?<=Avg total time   : )[\d.]+|(?<=Avg: )[\d.]+' "${log}" | head -1 || true)
    qps=$(grep -oP '(?<=QPS              : )[\d.]+|(?<=QPS: )[\d.]+' "${log}" | head -1 || true)

    peak_mem_kb=$(grep -oP "(?<=Maximum resident set size \(kbytes\): )\d+" "${timelog}" | head -1 || true)

    if [ -n "${peak_mem_kb:-}" ]; then
        peak_mem_mb=$(awk -v kb="${peak_mem_kb}" 'BEGIN { printf "%.2f", kb / 1024 }')
    else
        peak_mem_mb="N/A"
    fi

    recall="${recall:-N/A}"
    avg_time="${avg_time:-N/A}"
    qps="${qps:-N/A}"

    echo "${recall},${avg_time},${qps},${peak_mem_mb}"
}

idx_size_mb() {
    local idx="$1"

    if ls "${idx}"* >/dev/null 2>&1; then
        du -sm "${idx}"* 2>/dev/null | awk '{s += $1} END {print s + 0}'
    else
        echo "N/A"
    fi
}

run_one() {
    local clusters="$1"
    local ef="$2"

    local idx="${IDX_DIR}/sift_c${clusters}"
    local out="${OUTDIR}/sift_c${clusters}_ef${ef}.txt"
    local log="${OUTDIR}/logs/clusters${clusters}_ef${ef}.log"
    local timelog="${OUTDIR}/logs/clusters${clusters}_ef${ef}.time.log"

    echo
    echo "────────────────────────────────────────────────────────────"
    echo "SIFT cluster-ef sweep: clusters=${clusters}, ef=${ef}"
    echo "query       = ${QRY}"
    echo "groundtruth = ${GT}"
    echo "────────────────────────────────────────────────────────────"

    if [ -f "${idx}.kmidx" ]; then
        echo "[LOAD] Existing index: ${idx}"

        /usr/bin/time -v -o "${timelog}" \
        "${BIN}" "${VEC}" "${STR}" "${QRY}" \
            "${K}" "${clusters}" "${out}" "${MAX_ITER}" \
            ann "pq_m=${PQ_M}" "ef=${ef}" "load=${idx}" "gt=${GT}" \
            2>&1 | tee "${log}"
    else
        echo "[BUILD] New index: ${idx}"

        /usr/bin/time -v -o "${timelog}" \
        "${BIN}" "${VEC}" "${STR}" "${QRY}" \
            "${K}" "${clusters}" "${out}" "${MAX_ITER}" \
            ann "pq_m=${PQ_M}" "ef=${ef}" "save=${idx}" "gt=${GT}" \
            2>&1 | tee "${log}"
    fi

    local metrics recall avg_time qps peak_mem_mb size_mb

    metrics=$(extract_metrics "${log}" "${timelog}")
    IFS=, read -r recall avg_time qps peak_mem_mb <<< "${metrics}"

    size_mb=$(idx_size_mb "${idx}")

    echo "${DATASET},ann,clusters,${clusters},${ef},${recall},${avg_time},${qps},${peak_mem_mb},${size_mb}" >> "${CSV}"

    echo "     recall=${recall}%  avg_time=${avg_time}ms  QPS=${qps}  peak_mem=${peak_mem_mb}MB  idx_size=${size_mb}MB"
}

echo "════════════════════════════════════════════════════════════"
echo " SIFT cluster-number x ef sweep"
echo " clusters = ${CLUSTER_LIST[*]}"
echo " ef_list  = ${EF_LIST[*]}"
echo " pq_m     = ${PQ_M}"
echo " query    = ${QRY}"
echo " gt       = ${GT}"
echo " output   = ${OUTDIR}"
echo "════════════════════════════════════════════════════════════"

check_inputs

for C in "${CLUSTER_LIST[@]}"; do
    for EF in "${EF_LIST[@]}"; do
        run_one "${C}" "${EF}"
    done
done

echo
echo "════════════════════════════════════════════════════════════"
echo " Summary: SIFT cluster x ef sweep"
echo "════════════════════════════════════════════════════════════"

printf "  %-8s %-12s %-8s %8s %12s %10s %12s %12s\n" \
    "Dataset" "Clusters" "EF" "Recall%" "Avg(ms)" "QPS" "Mem(MB)" "Idx(MB)"

printf "  %-8s %-12s %-8s %8s %12s %10s %12s %12s\n" \
    "────────" "────────────" "────────" "────────" "────────────" "──────────" "────────────" "────────────"

while IFS=, read -r dataset method param clusters ef recall avg_time qps peak_mem_mb idx_size_mb; do
    [ "${dataset}" = "dataset" ] && continue

    printf "  %-8s %-12s %-8s %8s %12s %10s %12s %12s\n" \
        "${dataset}" "${clusters}" "${ef}" "${recall}" "${avg_time}" "${qps}" "${peak_mem_mb}" "${idx_size_mb}"
done < "${CSV}"

echo
echo "CSV  → ${CSV}"
echo "Logs → ${OUTDIR}/logs/"
echo "Idx  → ${IDX_DIR}/"
