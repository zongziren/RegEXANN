#!/usr/bin/env bash
# run_all.sh
# Run all dataset experiments in tmux (except sift and tripclick).
# Each dataset runs in its own tmux window; failures do not stop others.
# Usage: bash scripts/run_all.sh

SESSION="regexann"

# Datasets to run (excluding sift and tripclick)
SCRIPTS=(
    "scripts/run_arxiv.sh"
    "scripts/run_audio.sh"
    "scripts/run_gist.sh"
    "scripts/run_laion.sh"
    "scripts/run_msong.sh"
    "scripts/run_words.sh"
)

# Create tmux session (detached), or attach to existing
tmux new-session -d -s "${SESSION}" 2>/dev/null || true

for i in "${!SCRIPTS[@]}"; do
    SCRIPT="${SCRIPTS[$i]}"
    NAME=$(basename "${SCRIPT}" .sh | sed 's/run_//')

    if [ "$i" -eq 0 ]; then
        # Use the first window that already exists
        tmux rename-window -t "${SESSION}:0" "${NAME}"
        tmux send-keys -t "${SESSION}:${NAME}" "bash ${SCRIPT}; echo '=== ${NAME} done (exit \$?) ==='" Enter
    else
        # Create a new window for each subsequent dataset
        tmux new-window -t "${SESSION}" -n "${NAME}"
        tmux send-keys -t "${SESSION}:${NAME}" "bash ${SCRIPT}; echo '=== ${NAME} done (exit \$?) ==='" Enter
    fi
done

echo "════════════════════════════════════════════════════════════"
echo "  Launched ${#SCRIPTS[@]} experiments in tmux session '${SESSION}'"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Attach   : tmux attach -t ${SESSION}"
echo "  Switch   : Ctrl-b  then  n / p  (next/prev window)"
echo "  Select   : Ctrl-b  then  0-9"
echo "  Detach   : Ctrl-b  then  d"
echo "  Kill all : tmux kill-session -t ${SESSION}"
echo ""
echo "  Windows:"
for SCRIPT in "${SCRIPTS[@]}"; do
    NAME=$(basename "${SCRIPT}" .sh | sed 's/run_//')
    printf "    %-12s  %s\n" "${NAME}" "${SCRIPT}"
done