#!/usr/bin/env bash
# scripts/run_baselines.sh
# ─────────────────────────────────────────────────────────────────────────────
# RegExANN — 完整 baseline 对比脚本
#
# 对应论文中的全部评测方法：
#   groundtruth   精确全扫描（生成 ground truth）
#   ann           RegExANN（k-means + trigram + PQ）
#   prefilter     Pre-filter 基线
#   postfilter    Post-filter 基线（多个过采样倍数）
#   hier          两级层次 RegExANN
#
# 用法示例（与用户给的命令行风格一致）：
#
#   # siftsmall 数据集
#   DATASET=siftsmall \
#   VEC=dataset/siftsmall/siftsmall_vectors.fvecs \
#   STR=dataset/siftsmall/siftsmall_titles_clean.txt \
#   QRY=dataset/siftsmall/query.txt \
#   bash scripts/run_baselines.sh
#
#   # arxiv 数据集，自定义聚类数
#   DATASET=arxiv CLUSTERS=200 \
#   VEC=dataset/arxiv/vectors.fvecs \
#   STR=dataset/arxiv/strings.txt \
#   QRY=dataset/arxiv/queries.txt \
#   bash scripts/run_baselines.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 参数（支持环境变量覆盖）────────────────────────────────────────────────
DATASET="${DATASET:-siftsmall}"
VEC="${VEC:?'请设置 VEC=<vectors.fvecs>'}"
STR="${STR:?'请设置 STR=<strings.txt>'}"
QRY="${QRY:?'请设置 QRY=<query.txt>'}"
K="${K:-10}"
CLUSTERS="${CLUSTERS:-100}"
MAX_ITER="${MAX_ITER:-30}"
PQ_M="${PQ_M:-8}"
OUTDIR="${OUTDIR:-results/${DATASET}}"
BIN="${BIN:-./build/regann}"
EVAL="${EVAL:-./build/eval_recall}"

# Post-filter 过采样倍数列表（空格分隔）
OVERSAMPLE_LIST="${OVERSAMPLE_LIST:-10 20 50}"

# Hierarchical 参数
HIER_K0="${HIER_K0:-0}"      # 0 = 自动取 sqrt(CLUSTERS)
HIER_NPROBE="${HIER_NPROBE:-0}"  # 0 = 自动取 k0/2

mkdir -p "${OUTDIR}/idx"

echo "════════════════════════════════════════════════════════════════"
echo "  RegExANN Baseline Suite"
echo "  Dataset : ${DATASET}"
echo "  Vectors : ${VEC}"
echo "  Strings : ${STR}"
echo "  Queries : ${QRY}"
echo "  K=${K}  clusters=${CLUSTERS}  pq_m=${PQ_M}  max_iter=${MAX_ITER}"
echo "  Output  : ${OUTDIR}/"
echo "════════════════════════════════════════════════════════════════"

GT="${OUTDIR}/gt.txt"

# ── Step 1: Ground Truth ─────────────────────────────────────────────────────
echo ""
echo "[ 1/6 ] Ground truth (full-scan exact search)"
echo "────────────────────────────────────────────────────────────────"
"${BIN}" "${VEC}" "${STR}" "${QRY}" \
    "${K}" "${CLUSTERS}" "${GT}" "${MAX_ITER}" \
    groundtruth

# ── Step 2: RegExANN — build + save ─────────────────────────────────────────
echo ""
echo "[ 2/6 ] RegExANN — build index (k=${CLUSTERS}, pq_m=${PQ_M})"
echo "────────────────────────────────────────────────────────────────"
"${BIN}" "${VEC}" "${STR}" "${QRY}" \
    "${K}" "${CLUSTERS}" "${OUTDIR}/ann.txt" "${MAX_ITER}" \
    ann \
    "pq_m=${PQ_M}" \
    "save=${OUTDIR}/idx/${DATASET}" \
    "gt=${GT}"

# ── Step 3: RegExANN — load (验证序列化正确性) ───────────────────────────────
echo ""
echo "[ 3/6 ] RegExANN — load saved index"
echo "────────────────────────────────────────────────────────────────"
"${BIN}" "${VEC}" "${STR}" "${QRY}" \
    "${K}" "${CLUSTERS}" "${OUTDIR}/ann_loaded.txt" "${MAX_ITER}" \
    ann \
    "pq_m=${PQ_M}" \
    "load=${OUTDIR}/idx/${DATASET}" \
    "gt=${GT}"

# ── Step 4: Pre-filter baseline ──────────────────────────────────────────────
echo ""
echo "[ 4/6 ] Pre-filter baseline"
echo "────────────────────────────────────────────────────────────────"
"${BIN}" "${VEC}" "${STR}" "${QRY}" \
    "${K}" "${CLUSTERS}" "${OUTDIR}/prefilter.txt" "${MAX_ITER}" \
    prefilter \
    "gt=${GT}"

# ── Step 5: Post-filter baseline（多个 oversample 倍数）────────────────────
echo ""
echo "[ 5/6 ] Post-filter baseline (oversample = ${OVERSAMPLE_LIST})"
echo "────────────────────────────────────────────────────────────────"
for OV in ${OVERSAMPLE_LIST}; do
    echo "  oversample=${OV}"
    "${BIN}" "${VEC}" "${STR}" "${QRY}" \
        "${K}" "${CLUSTERS}" "${OUTDIR}/postfilter_ov${OV}.txt" "${MAX_ITER}" \
        postfilter \
        "oversample=${OV}" \
        "gt=${GT}"
done

# ── Step 6: Hierarchical RegExANN ───────────────────────────────────────────
echo ""
echo "[ 6/6 ] Hierarchical RegExANN (k0=${HIER_K0}, nprobe=${HIER_NPROBE})"
echo "────────────────────────────────────────────────────────────────"
"${BIN}" "${VEC}" "${STR}" "${QRY}" \
    "${K}" "${CLUSTERS}" "${OUTDIR}/hier.txt" "${MAX_ITER}" \
    hier \
    "k0=${HIER_K0}" \
    "nprobe=${HIER_NPROBE}" \
    "pq_m=${PQ_M}" \
    "gt=${GT}"

# ── Recall@K 汇总表 ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Recall@${K} Summary  [${DATASET}]"
echo "════════════════════════════════════════════════════════════════"
printf "  %-26s  %s\n" "Method" "Recall@${K}"
printf "  %-26s  %s\n" "──────────────────────────" "────────────"

recall_line() {
    local file="$1" label="$2"
    if [ -f "${file}" ]; then
        local r
        r=$("${EVAL}" --gt "${GT}" --pred "${file}" --K "${K}" 2>/dev/null \
            | grep "Mean" | awk '{print $(NF-1), $NF}')
        printf "  %-26s  %s\n" "${label}" "${r}"
    else
        printf "  %-26s  (not found)\n" "${label}"
    fi
}

recall_line "${OUTDIR}/ann.txt"              "RegExANN (build)"
recall_line "${OUTDIR}/ann_loaded.txt"       "RegExANN (load)"
recall_line "${OUTDIR}/prefilter.txt"        "Pre-filter"
for OV in ${OVERSAMPLE_LIST}; do
    recall_line "${OUTDIR}/postfilter_ov${OV}.txt" "Post-filter oversample=${OV}"
done
recall_line "${OUTDIR}/hier.txt"             "Hierarchical"

echo ""
echo "Output files: ${OUTDIR}/"
echo "Index files : ${OUTDIR}/idx/"
