#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

BIN=./exp/build/regann

K=10
CLUSTERS=500
MAX_ITER=30
PQ_M=8

DATASETS=(
  arxiv
  sift
  gist
  laion
  audio
  msong
  dbpedia
  words
)

EF_LIST=(10 20 50 100 200 500 1000 2000)

GLOBAL_CSV="results/regexann_ef_all_summary.csv"
mkdir -p results

if [[ ! -f "${GLOBAL_CSV}" ]]; then
    echo "dataset,method,param,param_value,recall_pct,avg_time_ms,qps,peak_mem_mb,idx_size_mb" > "${GLOBAL_CSV}"
fi

run_one() {
    local dataset="$1"
    local ef="$2"

    local OUTDIR="results/${dataset}"
    local IDX="${OUTDIR}/idx/${dataset}"

    local VEC="dataset/${dataset}/vectors.fvecs"
    local STR="dataset/${dataset}/strings.txt"
    local QRY="dataset/${dataset}/query.txt"
    local GT="dataset/${dataset}/groundtruth.txt"

    local CSV="${OUTDIR}/summary.csv"

    mkdir -p "${OUTDIR}/idx" "${OUTDIR}/logs"

    if [[ ! -f "${CSV}" ]]; then
        echo "method,param,param_value,recall_pct,avg_time_ms,qps,peak_mem_mb,idx_size_mb" > "${CSV}"
    fi

    local OUT="${OUTDIR}/ann_ef${ef}.txt"
    local LOG="${OUTDIR}/logs/ann_ef${ef}.log"
    local TIMELOG="${OUTDIR}/logs/ann_ef${ef}.time.log"

    echo
    echo "→ dataset=${dataset}, method=ann, ef=${ef}"

    if [[ -f "${IDX}.kmidx" ]]; then
        echo "  [INDEX] load=${IDX}"

        /usr/bin/time -v -o "${TIMELOG}" \
        "${BIN}" "${VEC}" "${STR}" "${QRY}" \
            "${K}" "${CLUSTERS}" "${OUT}" "${MAX_ITER}" \
            ann "pq_m=${PQ_M}" "ef=${ef}" "load=${IDX}" "gt=${GT}" \
            2>&1 | tee "${LOG}"
    else
        echo "  [INDEX] save=${IDX}"

        /usr/bin/time -v -o "${TIMELOG}" \
        "${BIN}" "${VEC}" "${STR}" "${QRY}" \
            "${K}" "${CLUSTERS}" "${OUT}" "${MAX_ITER}" \
            ann "pq_m=${PQ_M}" "ef=${ef}" "save=${IDX}" "gt=${GT}" \
            2>&1 | tee "${LOG}"
    fi

    local recall avg_time qps peak_mem_kb peak_mem_mb idx_size_mb

    recall=$(grep '\[EVAL\] Recall' "${LOG}" | grep -oP '[\d.]+(?= %)' | head -1 || true)
    avg_time=$(grep -oP '(?<=Avg total time   : )[\d.]+|(?<=Avg: )[\d.]+' "${LOG}" | head -1 || true)
    qps=$(grep -oP '(?<=QPS              : )[\d.]+|(?<=QPS: )[\d.]+' "${LOG}" | head -1 || true)

    peak_mem_kb=$(grep -oP "(?<=Maximum resident set size \(kbytes\): )\d+" "${TIMELOG}" | head -1 || true)
    if [[ -n "${peak_mem_kb:-}" ]]; then
        peak_mem_mb=$(awk -v kb="${peak_mem_kb}" 'BEGIN { printf "%.2f", kb / 1024 }')
    else
        peak_mem_mb="N/A"
    fi

    if [[ -d "${OUTDIR}/idx" ]]; then
        idx_size_mb=$(du -sm "${OUTDIR}/idx" | awk '{print $1}')
    else
        idx_size_mb="N/A"
    fi

    recall="${recall:-N/A}"
    avg_time="${avg_time:-N/A}"
    qps="${qps:-N/A}"
    peak_mem_mb="${peak_mem_mb:-N/A}"
    idx_size_mb="${idx_size_mb:-N/A}"

    echo "ann,ef,${ef},${recall},${avg_time},${qps},${peak_mem_mb},${idx_size_mb}" >> "${CSV}"
    echo "${dataset},ann,ef,${ef},${recall},${avg_time},${qps},${peak_mem_mb},${idx_size_mb}" >> "${GLOBAL_CSV}"

    echo "  recall=${recall}%  avg_time=${avg_time}ms  QPS=${qps}  peak_mem=${peak_mem_mb}MB  idx_size=${idx_size_mb}MB"
}

for DATASET in "${DATASETS[@]}"; do
    VEC="dataset/${DATASET}/vectors.fvecs"
    STR="dataset/${DATASET}/strings.txt"
    QRY="dataset/${DATASET}/query.txt"
    GT="dataset/${DATASET}/groundtruth.txt"

    echo
    echo "════════════════════════════════════════════════════════════"
    echo "Dataset: ${DATASET}"
    echo "vectors     = ${VEC}"
    echo "strings     = ${STR}"
    echo "query       = ${QRY}"
    echo "groundtruth = ${GT}"
    echo "════════════════════════════════════════════════════════════"

    if [[ ! -f "${VEC}" ]]; then
        echo "[SKIP] Missing vectors: ${VEC}"
        continue
    fi

    if [[ ! -f "${STR}" ]]; then
        echo "[SKIP] Missing strings: ${STR}"
        continue
    fi

    if [[ ! -f "${QRY}" ]]; then
        echo "[SKIP] Missing query: ${QRY}"
        continue
    fi

    if [[ ! -f "${GT}" ]]; then
        echo "[SKIP] Missing groundtruth: ${GT}"
        continue
    fi

    for EF in "${EF_LIST[@]}"; do
        run_one "${DATASET}" "${EF}"
    done

    echo
    echo "[DONE] ${DATASET}"
    echo "summary = results/${DATASET}/summary.csv"
done

echo
echo "════════════════════════════════════════════════════════════"
echo "[ALL DONE]"
echo "Global summary = ${GLOBAL_CSV}"
echo "════════════════════════════════════════════════════════════"