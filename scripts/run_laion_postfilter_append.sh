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

# 如果 summary.csv 不存在，就创建表头
if [ ! -f "${CSV}" ]; then
    echo "method,param,param_value,recall_pct,avg_time_ms,qps,peak_mem_mb,idx_size_mb" > "${CSV}"
fi

# ------------------------------------------------------------
# Replace if same method,param,param_value exists;
# otherwise append after existing postfilter/oversample rows.
# ------------------------------------------------------------
upsert_csv_row() {
    local new_row="$1"
    local method="$2"
    local param="$3"
    local pval="$4"

    python3 - "$CSV" "$new_row" "$method" "$param" "$pval" <<'PY'
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
new_row = sys.argv[2]
method = sys.argv[3]
param = sys.argv[4]
pval = sys.argv[5]

lines = csv_path.read_text().splitlines()

if not lines:
    lines = ["method,param,param_value,recall_pct,avg_time_ms,qps,peak_mem_mb,idx_size_mb"]

header = lines[0]
body = lines[1:]

replaced = False
new_body = []

for line in body:
    parts = line.split(",")
    if len(parts) >= 3 and parts[0] == method and parts[1] == param and parts[2] == pval:
        new_body.append(new_row)
        replaced = True
    else:
        new_body.append(line)

if not replaced:
    # 插入到最后一个 postfilter,oversample 后面
    insert_pos = None
    for i, line in enumerate(new_body):
        parts = line.split(",")
        if len(parts) >= 2 and parts[0] == method and parts[1] == param:
            insert_pos = i + 1

    if insert_pos is None:
        new_body.append(new_row)
    else:
        new_body.insert(insert_pos, new_row)

csv_path.write_text("\n".join([header] + new_body) + "\n")
PY
}

run() {
    local method="$1"
    local param="$2"
    local param_val="$3"
    local out="$4"
    shift 4
    local extra_args=("$@")

    local log="${OUTDIR}/logs/${method}_${param}${param_val}.log"
    local timelog="${OUTDIR}/logs/${method}_${param}${param_val}.time.log"

    echo "  → ${method}  ${param}=${param_val}"

    /usr/bin/time -v -o "${timelog}" "${BIN}" "${VEC}" "${STR}" "${QRY}" \
        "${K}" "${CLUSTERS}" "${out}" "${MAX_ITER}" \
        "${method}" "${extra_args[@]}" "gt=${GT}" \
        2>&1 | tee "${log}"

    local recall avg_time qps peak_mem_kb peak_mem_mb idx_size_mb row

    recall=$(grep '\[EVAL\] Recall' "${log}" | grep -oP '[\d.]+(?= %)' | head -1 || true)
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

    recall="${recall:-N/A}"
    avg_time="${avg_time:-N/A}"
    qps="${qps:-N/A}"
    peak_mem_mb="${peak_mem_mb:-N/A}"
    idx_size_mb="${idx_size_mb:-N/A}"

    row="${method},${param},${param_val},${recall},${avg_time},${qps},${peak_mem_mb},${idx_size_mb}"

    upsert_csv_row "${row}" "${method}" "${param}" "${param_val}"

    echo "     recall=${recall}%  avg_time=${avg_time}ms  QPS=${qps}  peak_mem=${peak_mem_mb}MB  idx_size=${idx_size_mb}MB"
    echo "     updated CSV: ${CSV}"
    echo ""
}

echo "[ LAION ] Post-filter oversample sweep: 10 20 50 100"

for OV in 10 20 50 100; do
    run postfilter oversample "${OV}" "${OUTDIR}/postfilter_ov${OV}.txt" \
        "oversample=${OV}" "max_expansion=${OV}"
done

echo "════════════════════════════════════════════════════════════"
echo "  Updated Recall / Speed Summary [laion]"
echo "════════════════════════════════════════════════════════════"

printf "  %-38s  %8s  %12s  %10s  %12s  %12s\n" \
    "Method" "Recall%" "Avg time(ms)" "QPS" "Mem(MB)" "Idx(MB)"

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
