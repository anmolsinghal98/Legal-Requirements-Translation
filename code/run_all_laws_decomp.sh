#!/usr/bin/env bash
# Run run_qwen_decomposition.py over the test set.
# Three passes per law, temperature 0.5, output -> qwen_results.
#
# Run from code/.
#
# Usage:
#   bash run_all_laws_decomp.sh             # run everything
#   bash run_all_laws_decomp.sh COPPA MS    # run only listed law tags

set -u  # don't set -e: one law's outage shouldn't kill the rest

OUTPUT_DIR="qwen_results"
PASSES=3
TEMP=0.5
MODEL="Qwen/Qwen3.5-397B-A17B"
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
    log="logs/${tag}-qwen3p5-decomp.log"
    echo "==================================================================="
    echo "[$(date '+%H:%M:%S')] Decomp run: $tag  -> $log"
    echo "==================================================================="

    python -u run_qwen_decomposition.py \
        --test "$test_file" \
        --output-dir "$OUTPUT_DIR" \
        --law-tag "$tag" \
        --passes "$PASSES" \
        --temperature "$TEMP" \
        --model "$MODEL" \
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
    echo "Re-run just those with: bash run_all_laws_decomp.sh ${failed[*]}"
fi
echo "==================================================================="
