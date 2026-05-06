#!/usr/bin/env bash
# Run run_codex_decomposition.py over the test set.
# Three passes per law, output -> codex_results/.
#
# Run from code/.
#
# Usage:
#   bash run_all_laws_codex.sh             # run everything
#   bash run_all_laws_codex.sh COPPA MS    # run only listed law tags

set -u

OUTPUT_DIR="codex_results"
PASSES=3
MODEL="gpt-5.3-codex"
EFFORT="low"
TEST_DIR="../test files"

declare -a LAWS=(COPPA MS OR UT VA VT WI)

FILTER=("$@")
should_run() {
    local tag="$1"
    if [[ ${#FILTER[@]} -eq 0 ]]; then return 0; fi
    for f in "${FILTER[@]}"; do
        [[ "$f" == "$tag" ]] && return 0
    done
    return 1
}

mkdir -p "$OUTPUT_DIR" logs

failed=()
for tag in "${LAWS[@]}"; do
    if ! should_run "$tag"; then continue; fi

    test_file="${TEST_DIR}/${tag}.csv"
    log="logs/${tag}-codex-decomp.log"
    echo "==================================================================="
    echo "[$(date '+%H:%M:%S')] Codex decomp run: $tag  -> $log"
    echo "==================================================================="

    python -u run_codex_decomposition.py \
        --test "$test_file" \
        --output-dir "$OUTPUT_DIR" \
        --law-tag "$tag" \
        --passes "$PASSES" \
        --model "$MODEL" \
        --reasoning-effort "$EFFORT" \
        2>&1 | tee "$log"

    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "!! $tag FAILED (exit $rc) — continuing"
        failed+=("$tag")
    fi
done

echo
echo "==================================================================="
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All laws completed."
else
    echo "Completed with failures: ${failed[*]}"
    echo "Re-run just those with: bash run_all_laws_codex.sh ${failed[*]}"
fi
echo "==================================================================="
