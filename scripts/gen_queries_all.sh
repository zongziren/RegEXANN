#!/usr/bin/env bash
# gen_queries_all.sh
# Generate 100 query files for all 8 datasets.
# Usage: bash scripts/gen_queries_all.sh [style]
#   style: substring | suffix | alternation | wildcard | mixed (default: mixed)

set -euo pipefail

GEN=dataset/gen_query.py
STYLE="${{1:-mixed}}"
SEED="${{SEED:-42}}"
NUM_QUERIES=100

run_gen() {{
    local name="$1" fvecs="$2" strings="$3"
    local out="dataset/${{name}}/query.txt"
    echo "  → ${{name}}  (${{NUM_QUERIES}} queries, style=${{STYLE}})"
    mkdir -p "dataset/${{name}}"
    python3 "${{GEN}}" \
        --fvecs       "${{fvecs}}" \
        --strings     "${{strings}}" \
        --output      "${{out}}" \
        --num_queries "${{NUM_QUERIES}}" \
        --style       "${{STYLE}}" \
        --seed        "${{SEED}}"
    echo "     saved → ${{out}}"
    echo ""
}}

echo "════════════════════════════════════════════════════════════"
echo "  Generating 100 queries for all 8 datasets  (style=${{STYLE}})"
echo "════════════════════════════════════════════════════════════"
echo ""

run_gen  arxiv      dataset/arxiv/vectors.fvecs               dataset/arxiv/strings.txt
run_gen  words      dataset/words/vectors.fvecs               dataset/words/strings.txt
run_gen  tripclick  dataset/tripclick/vectors.fvecs           dataset/tripclick/strings.txt
run_gen  sift       dataset/sift/sift_vectors.fvecs           dataset/sift/sift_titles_clean.txt
run_gen  gist       dataset/gist/vectors.fvecs                dataset/gist/strings.txt
run_gen  laion      dataset/laion/vectors.fvecs               dataset/laion/strings.txt
run_gen  msong      dataset/msong/vectors.fvecs               dataset/msong/strings.txt
run_gen  audio      dataset/audio/vectors.fvecs               dataset/audio/strings.txt

echo "════════════════════════════════════════════════════════════"
echo "  Done. 100 queries → dataset/<name>/query.txt"
echo "════════════════════════════════════════════════════════════"
