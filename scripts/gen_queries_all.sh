#!/usr/bin/env bash
set -euo pipefail

GEN=dataset/gen_query.py
PYTHON=/usr/bin/python3

STYLE="${1:-all}"
N_QUERIES="${N_QUERIES:-100}"
TOPK="${TOPK:-10}"
SEED="${SEED:-42}"

MIN_SEL="${MIN_SEL:-0.01}"
MAX_SEL="${MAX_SEL:-0.10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10000}"

run_gen() {
    local name="$1"
    local title_file="$2"
    local fvec_file="$3"

    local out_q="dataset/${name}/query.txt"
    local out_gt="dataset/${name}/groundtruth.txt"

    echo "  → ${name}"
    echo "     queries=${N_QUERIES}, style=${STYLE}, selectivity=[${MIN_SEL}, ${MAX_SEL}], max_attempts=${MAX_ATTEMPTS}"

    if [[ ! -f "${title_file}" ]]; then
        echo "     [SKIP] missing title file: ${title_file}"
        echo ""
        return 0
    fi

    if [[ ! -f "${fvec_file}" ]]; then
        echo "     [SKIP] missing vector file: ${fvec_file}"
        echo ""
        return 0
    fi

    mkdir -p "dataset/${name}"

    "${PYTHON}" "${GEN}" \
        "${title_file}" \
        "${fvec_file}" \
        "${out_q}" \
        "${out_gt}" \
        --n_queries "${N_QUERIES}" \
        --topk "${TOPK}" \
        --style "${STYLE}" \
        --seed "${SEED}" \
        --min_selectivity "${MIN_SEL}" \
        --max_selectivity "${MAX_SEL}" \
        --max_attempts "${MAX_ATTEMPTS}"

    echo "     queries      → ${out_q}"
    echo "     groundtruth  → ${out_gt}"
    echo ""
}

echo "════════════════════════════════════════════════════════════"
echo "  Generating ${N_QUERIES} queries for all 8 datasets"
echo "  style        = ${STYLE}"
echo "  topk         = ${TOPK}"
echo "  seed         = ${SEED}"
echo "  selectivity  = ${MIN_SEL} - ${MAX_SEL}"
echo "  max_attempts = ${MAX_ATTEMPTS}"
echo "════════════════════════════════════════════════════════════"
echo ""

#run_gen arxiv   dataset/arxiv/strings.txt          dataset/arxiv/vectors.fvecs
#run_gen words   dataset/words/strings.txt          dataset/words/vectors.fvecs
run_gen dbpedia dataset/dbpedia/strings.txt        dataset/dbpedia/vectors.fvecs
#run_gen sift    dataset/sift/strings.txt           dataset/sift/vectors.fvecs
#run_gen gist    dataset/gist/strings.txt           dataset/gist/vectors.fvecs
#run_gen laion   dataset/laion/strings.txt          dataset/laion/vectors.fvecs
#run_gen msong   dataset/msong/strings.txt          dataset/msong/vectors.fvecs
#run_gen audio   dataset/audio/strings.txt          dataset/audio/vectors.fvecs

echo "════════════════════════════════════════════════════════════"
echo "  Done."
echo "  Queries     → dataset/<name>/query.txt"
echo "  Groundtruth → dataset/<name>/groundtruth.txt"
echo "════════════════════════════════════════════════════════════"
