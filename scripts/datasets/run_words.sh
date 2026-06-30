#!/usr/bin/env bash
set -euo pipefail

BIN=./exp/build/regann
VEC=dataset/words/vectors.fvecs
STR=dataset/words/strings.txt
QRY=dataset/words/query.txt
GT=dataset/words/groundtruth.txt
K=10
CLUSTERS=50
MAX_ITER=30
OUTDIR=results/words
IDX="${OUTDIR}/idx/words"

mkdir -p "${OUTDIR}/idx" "${OUTDIR}/logs"

CSV="${OUTDIR}/summary.csv"
echo "method,param,param_value,recall_pct,avg_time_ms,qps,peak_mem_mb,idx_size_mb,t1_trigram_parse_ms,t2_cluster_lookup_ms,t3_pq_scan_ms,t4_regex_verify_ms,t5_rerank_ms" > "${CSV}"

# ── Helper ────────────────────────────────────────────────────────────────────
run() {
    local method="$1" param="$2" param_val="$3" out="$4"
    shift 4
    local extra_args=("$@")
    local log="${OUTDIR}/logs/${method}_${param}${param_val}.log"
    local timelog="${OUTDIR}/logs/${method}_${param}${param_val}.time.log"

    echo "  → ${method}  ${param}=${param_val}"

    /usr/bin/time -v -o "${timelog}" "${BIN}" "${VEC}" "${STR}" "${QRY}" \
        "${K}" "${CLUSTERS}" "${out}" "${MAX_ITER}" \
        "${method}" "${extra_args[@]}" "gt=${GT}" \
        2>&1 | tee "${log}"

    local recall avg_time qps peak_mem_kb peak_mem_mb idx_size_mb
    recall=$(grep '\[EVAL\] Recall' "${log}"  | grep -oP '[\d.]+(?= %)' | head -1 || true)
    avg_time=$(grep -oP '(?<=Avg total time   : )[\d.]+|(?<=Avg: )[\d.]+' "${log}" | head -1 || true)
    qps=$(grep -oP '(?<=QPS              : )[\d.]+|(?<=QPS: )[\d.]+' "${log}" | head -1 || true)
    peak_mem_kb=$(grep -oP "(?<=Maximum resident set size \(kbytes\): )\d+" "${timelog}" | head -1 || true)
    if [ -n "${peak_mem_kb:-}" ]; then
        peak_mem_mb=$(awk -v kb="${peak_mem_kb}" 'BEGIN { printf "%.2f", kb / 1024 }')
    else
        peak_mem_mb="N/A"
    fi
    if [ -d "${OUTDIR}/idx" ]; then
        idx_size_mb=$(du -sm "${OUTDIR}/idx" | awk '{print $1}')
    else
        idx_size_mb="N/A"
    fi

    # 4-stage pipeline breakdown (printed only for method=ann; N/A for other methods)
    local t1 t2 t3 t4_regex t5_rerank
    t1=$(grep -oP '(?<=\(1\) trigram parse  : )[\d.]+' "${log}" || true)
    t2=$(grep -oP '(?<=\(2\) cluster lookup : )[\d.]+' "${log}" || true)
    t3=$(grep -oP '(?<=\(3\) PQ candidate scan : )[\d.]+' "${log}" || true)
    t4_regex=$(grep -oP '(?<=regex=)[\d.]+' "${log}" || true)
    t5_rerank=$(grep -oP '(?<=rerank=)[\d.]+' "${log}" || true)

    recall="${recall:-N/A}"
    avg_time="${avg_time:-N/A}"
    qps="${qps:-N/A}"
    peak_mem_mb="${peak_mem_mb:-N/A}"
    idx_size_mb="${idx_size_mb:-N/A}"
    t1="${t1:-N/A}"; t2="${t2:-N/A}"; t3="${t3:-N/A}"
    t4_regex="${t4_regex:-N/A}"; t5_rerank="${t5_rerank:-N/A}"

    echo "${method},${param},${param_val},${recall},${avg_time},${qps},${peak_mem_mb},${idx_size_mb},${t1},${t2},${t3},${t4_regex},${t5_rerank}" >> "${CSV}"
    echo "     recall=${recall}%  avg_time=${avg_time}ms  QPS=${qps}  peak_mem=${peak_mem_mb}MB  idx_size=${idx_size_mb}MB  [t1=${t1} t2=${t2} t3=${t3} t4_regex=${t4_regex} t5_rerank=${t5_rerank}]"
    echo ""
}

# # ── 1. Ground Truth ───────────────────────────────────────────────────────────
# echo "[ 1 ] Ground truth"
# if [ -f "${GT}" ]; then
#     echo "  (skipped — ${GT} already exists)"
# else
#     /usr/bin/time -v -o "${timelog}" "${BIN}" "${VEC}" "${STR}" "${QRY}" \
#         "${K}" "${CLUSTERS}" "${GT}" "${MAX_ITER}" \
#         groundtruth 2>&1 | tee "${OUTDIR}/logs/groundtruth.log"
# fi
# echo ""

# ── 2. RegExANN — ef sweep (6 values) ────────────────────────────────────────
echo "[ 2 ] RegExANN (ef sweep: 10 20 30 50 75 100)"
for EF in 10 20 30 50 75 100 200 500; do
    OUT="${OUTDIR}/ann_ef${EF}.txt"
    if [ ! -f "${IDX}.kmidx" ]; then
        run ann ef "${EF}" "${OUT}" pq_m=8 "ef=${EF}" "save=${IDX}"
    else
        run ann ef "${EF}" "${OUT}" pq_m=8 "ef=${EF}" "load=${IDX}"
    fi
done

# ── 3. Pre-filter — sample_ratio sweep (6 values) ────────────────────────────
echo "[ 3 ] Pre-filter (sample_ratio sweep: 1.0 0.8 0.6 0.4 0.2 0.1)"
for SR in 1.0 0.8 0.6 0.4 0.2 0.1; do
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
echo "  Recall / Speed Summary [words]"
echo "  clusters=50  K=10  queries=100"
echo "════════════════════════════════════════════════════════════"
printf "  %-38s  %8s  %12s  %10s  %12s  %12s\n" "Method" "Recall%" "Avg time(ms)" "QPS" "Mem(MB)" "Idx(MB)"
printf "  %-38s  %8s  %12s  %10s  %12s  %12s\n" \
    "──────────────────────────────────────" "────────" "────────────" "──────────" "────────────" "────────────"
while IFS=, read -r method param pval recall avg_time qps peak_mem_mb idx_size_mb; do
    [ "${method}" = "method" ] && continue
    printf "  %-38s  %8s  %12s  %10s  %12s  %12s\n" \
        "${method} ${param}=${pval}" "${recall}" "${avg_time}" "${qps}" "${peak_mem_mb}" "${idx_size_mb}"
done < "${CSV}"
echo ""
echo "CSV  → ${CSV}"
echo "Logs → ${OUTDIR}/logs/"
