#!/usr/bin/env bash
set -euo pipefail

BIN=./exp/build/regann
VEC=dataset/siftsmall/siftsmall_vectors.fvecs
STR=dataset/siftsmall/siftsmall_titles_clean.txt
QRY=dataset/siftsmall/query.txt
K=10
CLUSTERS=100
MAX_ITER=30
OUTDIR=results/siftsmall
IDX="${OUTDIR}/idx/siftsmall"

mkdir -p "${OUTDIR}/idx" "${OUTDIR}/logs"

CSV="${OUTDIR}/summary.csv"
echo "method,param,param_value,recall_pct,avg_time_ms,qps" > "${CSV}"

# ── Helper: run binary, tee log, extract metrics, append to CSV ──────────────
run() {
    local method="$1" param="$2" param_val="$3" out="$4"
    shift 4
    local extra_args=("$@")
    local log="${OUTDIR}/logs/${method}_${param}${param_val}.log"

    echo "  → ${method}  ${param}=${param_val}"

    "${BIN}" "${VEC}" "${STR}" "${QRY}" \
        "${K}" "${CLUSTERS}" "${out}" "${MAX_ITER}" \
        "${method}" "${extra_args[@]}" "gt=${OUTDIR}/gt.txt" \
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

# ── 1. Ground Truth ───────────────────────────────────────────────────────────
echo "[ 1 ] Ground truth"
if [ -f "${OUTDIR}/gt.txt" ]; then
    echo "  (skipped — ${OUTDIR}/gt.txt already exists)"
else
    "${BIN}" "${VEC}" "${STR}" "${QRY}" \
        "${K}" "${CLUSTERS}" "${OUTDIR}/gt.txt" "${MAX_ITER}" \
        groundtruth 2>&1 | tee "${OUTDIR}/logs/groundtruth.log"
fi
echo ""

# ── 2. RegExANN — ef sweep ────────────────────────────────────────────────────
echo "[ 2 ] RegExANN (ef sweep)"
for EF in ${K} 20 50 100; do
    OUT="${OUTDIR}/ann_ef${EF}.txt"
    if [ ! -f "${IDX}.kmidx" ]; then
        run ann ef "${EF}" "${OUT}" pq_m=8 "ef=${EF}" "save=${IDX}"
    else
        run ann ef "${EF}" "${OUT}" pq_m=8 "ef=${EF}" "load=${IDX}"
    fi
done

# ── 3. RegExANN — nprobe sweep (ef=20 固定) ───────────────────────────────────
echo "[ 3 ] RegExANN (nprobe sweep, ef=20)"
for NP in 5 10 20 50; do
    run ann nprobe "${NP}" "${OUTDIR}/ann_nprobe${NP}.txt" \
        pq_m=8 ef=20 "nprobe=${NP}" "load=${IDX}"
done

# ── 4. Pre-filter — sample_ratio sweep ───────────────────────────────────────
echo "[ 4 ] Pre-filter (sample_ratio sweep)"
for SR in 1.0 0.5 0.2; do
    SR_TAG=$(echo "${SR}" | tr '.' 'p')
    run prefilter sample_ratio "${SR_TAG}" "${OUTDIR}/prefilter_sr${SR_TAG}.txt" \
        "sample_ratio=${SR}"
done

# ── 5. Post-filter — oversample sweep ────────────────────────────────────────
echo "[ 5 ] Post-filter (oversample sweep)"
for OV in 10 20 50; do
    run postfilter oversample "${OV}" "${OUTDIR}/postfilter_ov${OV}.txt" \
        "oversample=${OV}" "max_expansion=${OV}"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  Recall / Speed Summary [siftsmall]"
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
