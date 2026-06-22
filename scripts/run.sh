#!/usr/bin/env bash
# run_all_serial.sh
# Run all dataset experiments sequentially (except sift and tripclick).
# Failures do not stop others.
# Usage: bash scripts/run_all_serial.sh

# Datasets to run (excluding sift and tripclick)
SCRIPTS=(
    "scripts/run_arxiv.sh"
    "scripts/run_audio.sh"
    "scripts/run_gist.sh"
    "scripts/run_laion.sh"
    "scripts/run_msong.sh"
    "scripts/run_words.sh"
    "scripts/run_sift.sh"
    "scripts/run_dbpedia.sh"
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