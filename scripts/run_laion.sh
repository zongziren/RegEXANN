#!/usr/bin/env bash
set -euo pipefail

BIN=./exp/build/regann
VEC=dataset/laion/vectors.fvecs
STR=dataset/laion/strings.txt
QRY=dataset/laion/query.txt
GT=dataset/laion/groundtruth.txt
K=10
CLUSTERS=500
MAX_ITER=30
OUTDIR=results/laion
IDX="${OUTDIR}/idx/laion"

mkdir -p "${OUTDIR}/idx" "${OUTDIR}/logs"

CSV="${OUTDIR}/summary.csv"
echo "method,param,param_value,recall_pct,avg_time_ms,qps" > "${CSV}"

# ── Helper ────────────────────────────────────────────────────────────────────
run() {
    local method="$1" param="$2" param_val="$3" out="$4"
    shift 4
    local extra_args=("$@")
    local log="${OUTDIR}/logs/${method}_${param}${param_val}.log"

    echo "  → ${method}  ${param}=${param_val}"

    "${BIN}" "${VEC}" "${STR}" "${QRY}" \
        "${K}" "${CLUSTERS}" "${out}" "${MAX_ITER}" \
        "${method}" "${extra_args[@]}" "gt=${GT}" \
        2>&1 | tee "${log}"

    local recall avg_time qps
    recall=$(grep '\[EVAL\] Recall' "${log}"  | grep -oP '[\d.]+(?= %)' | head -1 || true)
    avg_time=$(grep -oP '(?<=Avg total time   : )[\d.]+|(?<=Avg: )[\d.]+' "${log}" | head -1 || true)
    qps=$(grep -oP '(?<=QPS              : )[\d.]+|(?<=QPS: )[\d.]+' "${log}" | head -1 || true)

    recall="${recall:-N/A}"
    avg_time="${avg_time:-N/A}"
    qps="${qps:-N/A}"

    echo "${method},${param},${param_val},${recall},${avg_time},${qps}" >> "${CSV}"
    echo "     recall=${recall}%  avg_time=${avg_time}ms  QPS=${qps}"
    echo ""
}

# # ── 1. Ground Truth ───────────────────────────────────────────────────────────
# echo "[ 1 ] Ground truth"
# if [ -f "${GT}" ]; then
#     echo "  (skipped — ${GT} already exists)"
# else
#     "${BIN}" "${VEC}" "${STR}" "${QRY}" \
#         "${K}" "${CLUSTERS}" "${GT}" "${MAX_ITER}" \
#         groundtruth 2>&1 | tee "${OUTDIR}/logs/groundtruth.log"
# fi
# echo ""

# ── 2. RegExANN — ef sweep (6 values) ────────────────────────────────────────
echo "[ 2 ] RegExANN (ef sweep: 10 20 30 50 75 100)"
for EF in 10 20 30 50 75 100 250 500; do
    OUT="${OUTDIR}/ann_ef${EF}.txt"
    if [ ! -f "${IDX}.kmidx" ]; then
        run ann ef "${EF}" "${OUT}" pq_m=8 "ef=${EF}" "save=${IDX}"
    else
        run ann ef "${EF}" "${OUT}" pq_m=8 "ef=${EF}" "load=${IDX}"
    fi
done

# ── 3. Pre-filter — sample_ratio sweep (6 values) ────────────────────────────
echo "[ 3 ] Pre-filter (sample_ratio sweep: 1.0 0.9 0.8 0.7 0.5 0.3)"
for SR in 1.0 0.9 0.8 0.7 0.5 0.3; do
    SR_TAG=$(echo "${SR}" | tr '.' 'p')
    run prefilter sample_ratio "${SR_TAG}" "${OUTDIR}/prefilter_sr${SR_TAG}.txt" \
        "sample_ratio=${SR}"
done

# ── 4. Post-filter — oversample sweep (6 values, max 1000) ───────────────────
echo "[ 4 ] Post-filter (oversample sweep: 10 100 1000 10000 100000)"
for OV in 10 100 1000 10000 100000; do
    run postfilter oversample "${OV}" "${OUTDIR}/postfilter_ov${OV}.txt" \
        "oversample=${OV}" "max_expansion=${OV}"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  Recall / Speed Summary [laion]"
echo "  clusters=500  K=10  queries=100"
echo "════════════════════════════════════════════════════════════"
printf "  %-38s  %8s  %12s  %10s\n" "Method" "Recall%" "Avg time(ms)" "QPS"
printf "  %-38s  %8s  %12s  %10s\n" \
    "──────────────────────────────────────" "────────" "────────────" "──────────"
while IFS=, read -r method param pval recall avg_time qps; do
    [ "${method}" = "method" ] && continue
    printf "  %-38s  %8s  %12s  %10s\n" \
        "${method} ${param}=${pval}" "${recall}" "${avg_time}" "${qps}"
done < "${CSV}"
echo ""
echo "CSV  → ${CSV}"
echo "Logs → ${OUTDIR}/logs/"
