"""
Run compilation, structural, semantic, and pass@k evaluation on a model's
output directory.

Filename pattern uses two placeholders:
  {law}  -> e.g. COPPA, MS, OR, UT, VA, VT, WI
  {pass} -> 1, 2, 3

Examples:
  # qwen single-shot
  python run_eval.py \
    --model-tag qwen_singleshot \
    --gen-dir ../qwen_results \
    --pattern '{law}-qwen3p5-397b-a17b-{pass}.csv' \
    --passes 3

  # qwen decomposition (step 3 only)
  python run_eval.py \
    --model-tag qwen_decomp \
    --gen-dir ../qwen_results/decomp \
    --pattern '{law}-qwen3p5-397b-a17b-step3-{pass}.csv' \
    --passes 3

  # codex decomposition
  python run_eval.py \
    --model-tag codex_decomp \
    --gen-dir ../codex_results \
    --pattern '{law}-gpt-5p3-codex-step3-{pass}.csv' \
    --passes 3

Outputs land in ./evaluation_output/<model_tag>/.
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from eval_lib import (
    TOTAL_SEMANTIC_TESTS,
    align_rows,
    build_compilation_df,
    build_semantic_df,
    build_structural_df,
    calculate_distribution,
    consolidate_pass_at_k,
    metrics_from_consolidated,
    normalize_gen_df,
    normalize_gt_df,
)

DEFAULT_LAWS = ["COPPA", "MS", "OR", "UT", "VA", "VT", "WI"]
# code/evaluation/run_eval.py -> code/evaluation -> code -> repo root -> test files
GT_DIR = Path(__file__).resolve().parent.parent.parent / "test files"


def evaluate_one(law: str, gen_dir: Path, pattern: str, passes: int,
                 out_dir: Path) -> dict:
    """Returns a flat metrics dict for the (model, law) pair."""
    gt_path = GT_DIR / f"{law}.csv"
    if not gt_path.exists():
        print(f"!! gt file missing: {gt_path}", file=sys.stderr)
        return {}
    gt_df = normalize_gt_df(pd.read_csv(gt_path,
                                        encoding="utf-8",
                                        encoding_errors="replace"))
    print(f"\n=== {law}  (gt rows: {len(gt_df)}) ===")

    semantic_dfs = []
    compile_pcts = []
    structural_means = []
    semantic_means = []

    for k in range(1, passes + 1):
        gen_path = gen_dir / pattern.format(law=law, **{"pass": k})
        if not gen_path.exists():
            print(f"  pass {k}: missing {gen_path.name} — skipping",
                  file=sys.stderr)
            continue
        gen_df_raw = pd.read_csv(gen_path, encoding="utf-8",
                                 encoding_errors="replace")
        gen_df = normalize_gen_df(gen_df_raw)
        gt_aln, gen_aln = align_rows(gt_df, gen_df)
        n = len(gt_aln)

        pass_dir = out_dir / law / f"pass{k}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        # Compilation
        cdf = build_compilation_df(gen_aln)
        cdf.to_csv(pass_dir / "compilation.csv", index=False)
        compile_pct = cdf["compiles"].sum() / n if n else 0.0
        compile_pcts.append(compile_pct)

        # Structural
        sdf = build_structural_df(gt_aln, gen_aln)
        sdf.to_csv(pass_dir / "structural.csv", index=False)
        struct_mean = sdf["Total Passed"].sum() / (n * 5) if n else 0.0
        structural_means.append(struct_mean)

        # Semantic
        mdf = build_semantic_df(gt_aln, gen_aln)
        mdf.to_csv(pass_dir / "semantic.csv", index=False)
        sem_acc = mdf["Accuracy"].mean() if n else 0.0
        semantic_means.append(sem_acc)
        semantic_dfs.append(mdf)

        print(f"  pass {k}:  compile={compile_pct:.3f}  "
              f"struct={struct_mean:.3f}  sem_acc={sem_acc:.3f}")

    if not semantic_dfs:
        return {}

    # Pass@k consolidation
    consolidated = consolidate_pass_at_k(semantic_dfs)
    consolidated.to_csv(out_dir / law / "passatk_consolidated.csv", index=False)

    dist = calculate_distribution(consolidated)
    pd.DataFrame(dist).to_csv(out_dir / law / "distribution.csv")

    overall = metrics_from_consolidated(consolidated)

    summary = {
        "law": law,
        "n_rows": overall["rows"],
        "passes_used": len(semantic_dfs),
        "compile_pass1": compile_pcts[0] if compile_pcts else 0.0,
        "compile_mean_passes": (sum(compile_pcts) / len(compile_pcts)
                                if compile_pcts else 0.0),
        "structural_pass1": structural_means[0] if structural_means else 0.0,
        "structural_mean_passes": (sum(structural_means) / len(structural_means)
                                   if structural_means else 0.0),
        "semantic_acc_pass1": semantic_means[0] if semantic_means else 0.0,
        "semantic_acc_consolidated": overall["mean_accuracy"],
        "semantic_recall_consolidated": overall["mean_recall"],
        "semantic_precision_consolidated": overall["mean_precision"],
        f"pass_at_{len(semantic_dfs)}": overall["pass_at_k"],
        "all_pass_count": overall["all_pass_count"],
    }
    print(f"  -> pass@{len(semantic_dfs)} = "
          f"{overall['pass_at_k']:.3f}  "
          f"acc={overall['mean_accuracy']:.3f}  "
          f"rec={overall['mean_recall']:.3f}  "
          f"pre={overall['mean_precision']:.3f}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-tag", required=True,
                   help="Subdir under evaluation_output/.")
    p.add_argument("--gen-dir", required=True, type=Path,
                   help="Directory containing the generated CSVs.")
    p.add_argument("--pattern", required=True,
                   help="Filename pattern with {law} and {pass} placeholders.")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--laws", nargs="+", default=DEFAULT_LAWS)
    p.add_argument("--out-root", type=Path,
                   default=Path(__file__).resolve().parent / "evaluation_output")
    args = p.parse_args()

    out_dir = args.out_root / args.model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model_tag={args.model_tag}  gen_dir={args.gen_dir}  "
          f"out={out_dir}")

    rows = []
    for law in args.laws:
        s = evaluate_one(law, args.gen_dir, args.pattern, args.passes, out_dir)
        if s:
            rows.append(s)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "summary.csv", index=False)
        print("\n--- Per-law summary ---")
        print(df.to_string(index=False))
        # macro-avg row
        avg = {"law": "AVG"}
        for c in df.columns:
            if c == "law":
                continue
            try:
                avg[c] = float(df[c].astype(float).mean())
            except Exception:
                avg[c] = ""
        with open(out_dir / "summary.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=df.columns.tolist())
            w.writerow(avg)
        print(f"\nWrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
