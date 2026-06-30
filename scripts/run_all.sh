#!/usr/bin/env bash
# Usage: bash scripts/run_all.sh

SCRIPTS=(
    "scripts/datasets/run_arxiv.sh"
    "scripts/datasets/run_audio.sh"
    "scripts/datasets/run_gist.sh"
    "scripts/datasets/run_laion.sh"
    "scripts/datasets/run_msong.sh"
    "scripts/datasets/run_words.sh"
    "scripts/datasets/run_sift.sh"
    "scripts/datasets/run_dbpedia.sh"
)

echo "════════════════════════════════════════════════════════════"
echo "  Running ${#SCRIPTS[@]} experiments sequentially"
echo "════════════════════════════════════════════════════════════"
echo ""

for SCRIPT in "${SCRIPTS[@]}"; do
    NAME=$(basename "${SCRIPT}" .sh | sed 's/run_//')

    echo "────────────────────────────────────────────────────────────"
    echo "  Start: ${NAME}"
    echo "  Script: ${SCRIPT}"
    echo "────────────────────────────────────────────────────────────"

    if bash "${SCRIPT}"; then
        echo "=== ${NAME} done (exit 0) ==="
    else
        STATUS=$?
        echo "=== ${NAME} failed (exit ${STATUS}) ==="
    fi

    echo ""
done

echo "════════════════════════════════════════════════════════════"
echo "  All experiments finished"
echo "════════════════════════════════════════════════════════════"