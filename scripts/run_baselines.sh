#!/usr/bin/env bash
# scripts/run_baselines.sh
# ─────────────────────────────────────────────────────────────────────────────
# RegExANN — full baseline comparison script
#
# Runs all evaluation methods and sweeps new recall/speed parameters:
#   groundtruth   Exact full scan (generates ground truth)
#   ann           RegExANN with ef sweep (ef controls candidate pool size)
#   prefilter     Pre-filter baseline with sample_ratio sweep
#   postfilter    Post-filter baseline with adaptive expansion + oversample sweep
#   hier          Two-level hierarchical RegExANN
#
# Usage examples:
#
#   # siftsmall dataset
#   DATASET=siftsmall \
#   VEC=dataset/siftsmall/siftsmall_vectors.fvecs \
#   STR=dataset/siftsmall/siftsmall_titles_clean.txt \
#   QRY=dataset/siftsmall/query.txt \
#   bash scripts/run_baselines.sh
#
#   # arxiv dataset with custom clusters
#   DATASET=arxiv CLUSTERS=200 \
#   VEC=dataset/arxiv/vectors.fvecs \
#   STR=dataset/arxiv/strings.txt \
#   QRY=dataset/arxiv/queries.txt \
#   bash scripts/run_baselines.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Parameters (override via environment variables) ───────────────────────────
DATASET="${DATASET:-siftsmall}"
VEC="${VEC:?'Set VEC=<vectors.fvecs>'}"
STR="${STR:?'Set STR=<strings.txt>'}"
QRY="${QRY:?'Set QRY=<query.txt>'}"
K="${K:-10}"
CLUSTERS="${CLUSTERS:-100}"
MAX_ITER="${MAX_ITER:-30}"
PQ_M="${PQ_M:-8}"
OUTDIR="${OUTDIR:-results/${DATASET}}"
BIN="${BIN:-./build/regann}"
EVAL="${EVAL:-./build/eval_recall}"

# ann: ef values to sweep (ef >= K; larger = higher recall, slower)
EF_LIST="${EF_LIST:-${K} 20 50 100}"

# postfilter: initial oversample factors to sweep
OVERSAMPLE_LIST="${OVERSAMPLE_LIST:-10 20 50}"

# postfilter: max_expansion cap (0 = unlimited adaptive expansion)
MAX_EXPANSION="${MAX_EXPANSION:-0}"

# prefilter: sample_ratio values to sweep (1.0 = original 100% recall)
SAMPLE_RATIOS="${SAMPLE_RATIOS:-1.0 0.5 0.2}"

# Hierarchical parameters
HIER_K0="${HIER_K0:-0}"      # 0 = auto: sqrt(CLUSTERS)
HIER_NPROBE="${HIER_NPROBE:-0}"  # 0 = auto: k0/2

mkdir -p "${OUTDIR}/idx"

echo "════════════════════════════════════════════════════════════════"
echo "  RegExANN Baseline Suite"
echo "  Dataset  : ${DATASET}"
echo "  Vectors  : ${VEC}"
echo "  Strings  : ${STR}"
echo "  Queries  : ${QRY}"
echo "  K=${K}  clusters=${CLUSTERS}  pq_m=${PQ_M}  max_iter=${MAX_ITER}"
echo "  Output   : ${OUTDIR}/"
echo "  EF sweep : ${EF_LIST}"
echo "  Oversample sweep : ${OVERSAMPLE_LIST}  max_expansion=${MAX_EXPANSION}"
echo "  Sample-ratio sweep : ${SAMPLE_RATIOS}"
echo "════════════════════════════════════════════════════════════════"

GT="${OUTDIR}/gt.txt"

# ── Step 1: Ground Truth ──────────────────────────────────────────────────────
echo ""
echo "[ 1/6 ] Ground truth (full-scan exact search)"
echo "────────────────────────────────────────────────────────────────"
"${BIN}" "${VEC}" "${STR}" "${QRY}" \
    "${K}" "${CLUSTERS}" "${GT}" "${MAX_ITER}" \
    groundtruth

# ── Step 2: RegExANN — build index + ef sweep ────────────────────────────────
echo ""
echo "[ 2/6 ] RegExANN — build index (k=${CLUSTERS}, pq_m=${PQ_M}), ef sweep: ${EF_LIST}"
echo "────────────────────────────────────────────────────────────────"
FIRST_EF=true
for EF in ${EF_LIST}; do
    OUT="${OUTDIR}/ann_ef${EF}.txt"
    echo "  ef=${EF}"
    if [ "${FIRST_EF}" = true ]; then
        # Build and save the index on first run
        "${BIN}" "${VEC}" "${STR}" "${QRY}" \
            "${K}" "${CLUSTERS}" "${OUT}" "${MAX_ITER}" \
            ann \
            "pq_m=${PQ_M}" \
            "ef=${EF}" \
            "save=${OUTDIR}/idx/${DATASET}" \
            "gt=${GT}"
        FIRST_EF=false
    else
        # Load saved index for subsequent ef values
        "${BIN}" "${VEC}" "${STR}" "${QRY}" \
            "${K}" "${CLUSTERS}" "${OUT}" "${MAX_ITER}" \
            ann \
            "pq_m=${PQ_M}" \
            "ef=${EF}" \
            "load=${OUTDIR}/idx/${DATASET}" \
            "gt=${GT}"
    fi
done

# ── Step 3: RegExANN — verify load (optional sanity check) ───────────────────
echo ""
echo "[ 3/6 ] RegExANN — load saved index (sanity check, ef=${K})"
echo "────────────────────────────────────────────────────────────────"
"${BIN}" "${VEC}" "${STR}" "${QRY}" \
    "${K}" "${CLUSTERS}" "${OUTDIR}/ann_loaded.txt" "${MAX_ITER}" \
    ann \
    "pq_m=${PQ_M}" \
    "ef=${K}" \
    "load=${OUTDIR}/idx/${DATASET}" \
    "gt=${GT}"

# ── Step 4: Pre-filter baseline — sample_ratio sweep ─────────────────────────
echo ""
echo "[ 4/6 ] Pre-filter baseline (sample_ratio sweep: ${SAMPLE_RATIOS})"
echo "────────────────────────────────────────────────────────────────"
for SR in ${SAMPLE_RATIOS}; do
    # Replace '.' with 'p' for filename safety (e.g. 0.5 -> 0p5)
    SR_TAG=$(echo "${SR}" | tr '.' 'p')
    OUT="${OUTDIR}/prefilter_sr${SR_TAG}.txt"
    echo "  sample_ratio=${SR}"
    "${BIN}" "${VEC}" "${STR}" "${QRY}" \
        "${K}" "${CLUSTERS}" "${OUT}" "${MAX_ITER}" \
        prefilter \
        "sample_ratio=${SR}" \
        "gt=${GT}"
done

# ── Step 5: Post-filter baseline — oversample sweep + adaptive expansion ──────
echo ""
echo "[ 5/6 ] Post-filter baseline (oversample sweep: ${OVERSAMPLE_LIST}, max_expansion=${MAX_EXPANSION})"
echo "────────────────────────────────────────────────────────────────"
for OV in ${OVERSAMPLE_LIST}; do
    OUT="${OUTDIR}/postfilter_ov${OV}.txt"
    echo "  oversample=${OV}  max_expansion=${MAX_EXPANSION}"
    "${BIN}" "${VEC}" "${STR}" "${QRY}" \
        "${K}" "${CLUSTERS}" "${OUT}" "${MAX_ITER}" \
        postfilter \
        "oversample=${OV}" \
        "max_expansion=${MAX_EXPANSION}" \
        "gt=${GT}"
done

# ── Step 6: Hierarchical RegExANN ─────────────────────────────────────────────
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

# ── Recall@K summary table ────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Recall@${K} Summary  [${DATASET}]"
echo "════════════════════════════════════════════════════════════════"
printf "  %-36s  %s\n" "Method" "Recall@${K}"
printf "  %-36s  %s\n" "────────────────────────────────────" "────────────"

recall_line() {
    local file="$1" label="$2"
    if [ -f "${file}" ]; then
        local r
        r=$("${EVAL}" --gt "${GT}" --pred "${file}" --K "${K}" 2>/dev/null \
            | grep "Mean" | awk '{print $(NF-1), $NF}') || r="(eval error)"
        printf "  %-36s  %s\n" "${label}" "${r}"
    else
        printf "  %-36s  (not found)\n" "${label}"
    fi
}

for EF in ${EF_LIST}; do
    recall_line "${OUTDIR}/ann_ef${EF}.txt"   "RegExANN ef=${EF}"
done
recall_line "${OUTDIR}/ann_loaded.txt"         "RegExANN (load, ef=${K})"

for SR in ${SAMPLE_RATIOS}; do
    SR_TAG=$(echo "${SR}" | tr '.' 'p')
    recall_line "${OUTDIR}/prefilter_sr${SR_TAG}.txt"  "Pre-filter sample_ratio=${SR}"
done

for OV in ${OVERSAMPLE_LIST}; do
    recall_line "${OUTDIR}/postfilter_ov${OV}.txt"  "Post-filter oversample=${OV} (adaptive)"
done

recall_line "${OUTDIR}/hier.txt"  "Hierarchical"

echo ""
echo "Output files : ${OUTDIR}/"
echo "Index files  : ${OUTDIR}/idx/"
