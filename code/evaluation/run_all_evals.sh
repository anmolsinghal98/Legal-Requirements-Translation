#!/usr/bin/env bash
# Run evaluation across all three model variants and produce a combined
# all_models_summary.csv at the top level.
set -u

cd "$(dirname "$0")"
mkdir -p evaluation_output logs

declare -a CONFIGS=(
    "qwen_decomp|../qwen_results|{law}-qwen3p5-397b-a17b-step3-{pass}.csv"
    "codex_decomp|../codex_results|{law}-gpt-5p3-codex-step3-{pass}.csv"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag dir pat <<< "$cfg"
    log="logs/${tag}-eval.log"
    echo "==================================================================="
    echo "[$(date '+%H:%M:%S')] $tag  ($dir)  -> $log"
    echo "==================================================================="
    python -u run_eval.py \
        --model-tag "$tag" \
        --gen-dir "$dir" \
        --pattern "$pat" \
        --passes 3 \
        --laws COPPA MS OR UT VA VT WI \
        2>&1 | tee "$log"
done

# Aggregate per-model summaries into one master CSV.
python - <<'PY'
import pandas as pd
from pathlib import Path
root = Path("evaluation_output")
frames = []
for sub in sorted(root.iterdir()):
    s = sub / "summary.csv"
    if not s.exists():
        continue
    df = pd.read_csv(s)
    df.insert(0, "model", sub.name)
    frames.append(df)
if frames:
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(root / "all_models_summary.csv", index=False)
    print(f"\nWrote {root / 'all_models_summary.csv'}  ({len(out)} rows)")
    print(out.to_string(index=False))
PY
