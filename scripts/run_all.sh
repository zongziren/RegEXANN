#!/usr/bin/env bash
# run_all.sh
# Run all 8 per-dataset experiments (scripts/datasets/run_<dataset>.sh)
# sequentially. Failures do not stop others.
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

# ── Pipeline-stage profiling aggregation (Figure 4b) ──────────────────────
# No longer re-runs regann: the 4-stage breakdown (t1..t5) is already
# captured into each dataset's summary.csv during the ef sweep above.
# This step just aggregates those columns across all 8 datasets into
# results/profile.csv. Set SKIP_PROFILE=1 to skip it.
if [[ "${SKIP_PROFILE:-0}" != "1" ]]; then
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Start: profile (aggregate pipeline-stage breakdown, Figure 4b)"
    echo "  Script: scripts/run_all_profile.sh"
    echo "────────────────────────────────────────────────────────────"
    if bash "scripts/run_all_profile.sh"; then
        echo "=== profile done (exit 0) ==="
    else
        STATUS=$?
        echo "=== profile failed (exit ${STATUS}) ==="
    fi
else
    echo ""
    echo "(SKIP_PROFILE=1 — skipping scripts/run_all_profile.sh)"
fi