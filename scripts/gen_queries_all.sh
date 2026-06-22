#!/usr/bin/env bash
# gen_queries_all.sh
# Generate 100 query files for all 8 datasets using gen_query.py
# Run from project root: bash scripts/gen_queries_all.sh [style]
#   style: substring | prefix | suffix | alternation | wildcard | mixed | all
#   default: mixed

set -euo pipefail

GEN=dataset/gen_query.py
STYLE="${1:-all}"
N_QUERIES=100
TOPK=10
SEED="${SEED:-42}"

run_gen() {
    local name="$1" title_file="$2" fvec_file="$3"
    local out_q="dataset/${name}/query.txt"
    local out_gt="dataset/${name}/groundtruth.txt"
    echo "  → ${name}  (${N_QUERIES} queries, style=${STYLE})"
    mkdir -p "dataset/${name}"
    python3 "${GEN}" \
        "${title_file}" \
        "${fvec_file}" \
        "${out_q}" \
        "${out_gt}" \
        --n_queries "${N_QUERIES}" \
        --topk      "${TOPK}" \
        --style     "${STYLE}" \
        --seed      "${SEED}"
    echo "     queries      → ${out_q}"
    echo "     groundtruth  → ${out_gt}"
    echo ""
}

echo "════════════════════════════════════════════════════════════"
echo "  Generating ${N_QUERIES} queries for all 8 datasets  (style=${STYLE})"
echo "════════════════════════════════════════════════════════════"
echo ""

#         name       title_file                                  fvec_file
# run_gen   arxiv      dataset/arxiv/strings.txt                  dataset/arxiv/vectors.fvecs
# run_gen   words      dataset/words/strings.txt                  dataset/words/vectors.fvecs
run_gen   dbpedia    dataset/dbpedia/strings.txt              dataset/dbpedia/vectors.fvecs
# run_gen   sift       dataset/sift/sift_titles_clean.txt         dataset/sift/sift_vectors.fvecs
# run_gen   gist       dataset/gist/strings.txt                   dataset/gist/vectors.fvecs
# run_gen   laion      dataset/laion/strings.txt                  dataset/laion/vectors.fvecs
# run_gen   msong      dataset/msong/strings.txt                  dataset/msong/vectors.fvecs
# run_gen   audio      dataset/audio/strings.txt                  dataset/audio/vectors.fvecs

echo "════════════════════════════════════════════════════════════"
echo "  Done."
echo "  Queries     → dataset/<name>/query.txt"
echo "  Groundtruth → dataset/<name>/groundtruth.txt"
echo "════════════════════════════════════════════════════════════"
