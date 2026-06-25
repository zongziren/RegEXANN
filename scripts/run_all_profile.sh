#!/usr/bin/env bash
# scripts/run_all_profile.sh
#
# Runs all 8 datasets, each with its own ef value (set in the DATASETS
# array below), loading each dataset's prebuilt index (load=, no save= —
# indices must already exist from a prior run of the corresponding
# run_<dataset>.sh script, which builds + saves them).
#
# Parses the 4-stage breakdown that main.cpp's run_ann prints to stdout
# (after the timing patch) and appends one row per dataset to a freshly
# created results/profile.csv.
#
# Does NOT modify or call any existing run_*.sh script.
#
# Usage:
#   bash scripts/run_all_profile.sh

set -euo pipefail

BIN=./exp/build/regann
K=10
MAX_ITER=30
PLACEHOLDER_CLUSTERS=1   # ignored by main.cpp in load= mode, just fills the positional arg

mkdir -p results
mkdir -p results/profile_logs
CSV=results/profile.csv
echo "dataset,ef,t1_trigram_parse_ms,t2_cluster_lookup_ms,t3_pq_scan_ms,t4_regex_verify_rerank_ms,t4_regex_only_ms,t5_rerank_only_ms,avg_total_ms,recall_pct" > "${CSV}"

# name | ef   (VEC/STR/QRY/GT paths follow the same dataset/<name>/
# layout as the existing run_<name>.sh scripts. No clusters column here:
# load= reads the cluster count from the saved index itself — the X
# positional arg is only used by main.cpp when actually building an
# index, so it's irrelevant in load mode and we just pass a placeholder.)
DATASETS=(
    "arxiv   2000"
    "audio   20"
    "dbpedia 1000"
    "gist    500"
    "laion   1000"
    "msong   100"
    "sift    50"
    "words   50"
)

for entry in "${DATASETS[@]}"; do
    read -r NAME EF <<< "${entry}"

    VEC="dataset/${NAME}/vectors.fvecs"
    STR="dataset/${NAME}/strings.txt"
    QRY="dataset/${NAME}/query.txt"
    GT="dataset/${NAME}/groundtruth.txt"
    IDX="results/${NAME}/idx/${NAME}"
    OUT="results/${NAME}/ann_profile_ef${EF}.txt"
    LOG="results/profile_logs/${NAME}.log"

    mkdir -p "results/${NAME}/idx"

    if [ ! -f "${IDX}.kmidx" ]; then
        echo "[SKIP] ${NAME}: no prebuilt index at ${IDX}.kmidx — run scripts/run_${NAME}.sh first."
        continue
    fi
    if [ ! -f "${VEC}" ] || [ ! -f "${QRY}" ]; then
        echo "[SKIP] ${NAME}: dataset files not found under dataset/${NAME}/"
        continue
    fi

    echo "→ ${NAME}  (ef=${EF}, load=${IDX})"

    # The clusters positional arg (here: PLACEHOLDER) is ignored by
    # main.cpp whenever load= succeeds — the real cluster count comes
    # from the saved index file itself.
    "${BIN}" "${VEC}" "${STR}" "${QRY}" "${K}" "${PLACEHOLDER_CLUSTERS}" "${OUT}" "${MAX_ITER}" \
        ann pq_m=8 "ef=${EF}" "load=${IDX}" "gt=${GT}" \
        > "${LOG}" 2>&1 || { echo "  [ERROR] ${NAME} failed — see ${LOG}"; continue; }

    t1=$(grep -oP '(?<=\(1\) trigram parse  : )[\d.]+' "${LOG}" || echo "")
    t2=$(grep -oP '(?<=\(2\) cluster lookup : )[\d.]+' "${LOG}" || echo "")
    t3=$(grep -oP '(?<=\(3\) PQ candidate scan : )[\d.]+' "${LOG}" || echo "")
    t4_total=$(grep -oP '(?<=\(4\) regex verify \+ rerank : )[\d.]+' "${LOG}" || echo "")
    t4_regex=$(grep -oP '(?<=regex=)[\d.]+' "${LOG}" || echo "")
    t5_rerank=$(grep -oP '(?<=rerank=)[\d.]+' "${LOG}" || echo "")
    avg_total=$(grep -oP '(?<=Avg total time   : )[\d.]+' "${LOG}" || echo "")
    recall=$(grep -oP '(?<=Recall@10 = )[\d.]+' "${LOG}" || echo "")

    t1="${t1:-N/A}"; t2="${t2:-N/A}"; t3="${t3:-N/A}"
    t4_total="${t4_total:-N/A}"; t4_regex="${t4_regex:-N/A}"; t5_rerank="${t5_rerank:-N/A}"
    avg_total="${avg_total:-N/A}"; recall="${recall:-N/A}"

    echo "${NAME},${EF},${t1},${t2},${t3},${t4_total},${t4_regex},${t5_rerank},${avg_total},${recall}" >> "${CSV}"
    echo "    t1=${t1}ms  t2=${t2}ms  t3=${t3}ms  t4(regex+rerank)=${t4_total}ms  total=${avg_total}ms  recall=${recall}%"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo " Done. Results → ${CSV}"
echo "════════════════════════════════════════════════════════════"
cat "${CSV}"
