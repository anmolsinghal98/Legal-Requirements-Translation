"""
Evaluate the intermediate tagging step of the Decomposition approach against
the manually-coded ground-truth tags.

The claim under test: tagging is a bottleneck for downstream example retrieval.
If predicted tags disagree systematically with the manually-coded ones, the
embedding-+-tag-overlap retrieval used in step 2 of the pipeline will pull
less-relevant demonstrations into the prompt, degrading code-generation
accuracy. This script measures that disagreement.

Predictions come from the GPT-5.1 Decomp run's step-1 CSVs in
code/intermediate-results/Decomp_gpt5/.
In the published notebook, the test set is tagged once before the pass loop;
the same tags are written into every pass's step-1 CSV. We verify this and
use pass 1 as the canonical prediction.

Metrics are computed at three layers:
  - per row: exact match, P/R/F1, Jaccard
  - per tag: support, TP/FP/FN, P/R/F1
  - per granularity (Statement / Phrase / Relation): exact-match rate +
    macro/micro P/R/F1 -- directly comparable to the human-vs-human IAA
    exact-match numbers reported in §3.2 of the paper
    (Statement 75.8%, Phrase 72.0%, Relation 57.7%).

Usage:
  python eval_tags.py
  python eval_tags.py --laws OR VT --passes 3
"""

import argparse
import ast
import csv
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Coding frame (paper Table 1) -- partitions the 17-tag inventory by granularity.
# ---------------------------------------------------------------------------
TAGS_BY_GRAN = {
    "Statement": [
        "#definition", "#exemption", "#obligation", "#permission",
        "#prohibition", "#penalty", "#information", "#continuation",
    ],
    "Phrase":    ["#exclusion", "#condition", "#reference"],
    "Relation":  ["#follows", "#refines", "#followed_by",
                  "#refined_by", "#exception", "#exception_to"],
}
ALL_TAGS = [t for ts in TAGS_BY_GRAN.values() for t in ts]
TAG_TO_GRAN = {t: g for g, ts in TAGS_BY_GRAN.items() for t in ts}

# code/evaluation/eval_tags.py -> code/evaluation -> code -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GT = REPO_ROOT / "Manual Coding - REJ Legal Translation Paper - Anmol - updated.csv"
DEFAULT_PRED_DIR = REPO_ROOT / "code" / "intermediate-results" / "Decomp_gpt5"
DEFAULT_LAWS = ["MS", "OR", "UT", "VA", "VT", "WI"]


# ---------------------------------------------------------------------------
# Text + tag normalisation
# ---------------------------------------------------------------------------
def norm_text(s: str) -> str:
    """Collapse whitespace and reverse MacRoman mojibake (¬ß for §, ¬† for
    NBSP, etc.) so the GT and prediction text columns join cleanly. The GT
    CSV was saved on macOS and went through a round-trip that decoded UTF-8
    bytes as MacRoman; encoding back to MacRoman and decoding as UTF-8
    reverses the corruption. Falls through cleanly on ASCII or other
    well-formed UTF-8."""
    if not isinstance(s, str):
        return ""
    try:
        s = s.encode("mac-roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_gt_cell(cell) -> set:
    """GT cell is comma-separated tags, possibly empty/NaN."""
    if not isinstance(cell, str) or not cell.strip():
        return set()
    parts = [p.strip().lower() for p in cell.split(",") if p.strip()]
    return {p for p in parts if p.startswith("#")}


def parse_pred_cell(cell) -> set:
    """Prediction cell is a stringified Python list of tags."""
    if isinstance(cell, list):
        return {str(t).strip().lower() for t in cell if str(t).strip()}
    if not isinstance(cell, str) or not cell.strip():
        return set()
    try:
        v = ast.literal_eval(cell)
        if isinstance(v, (list, tuple, set)):
            return {str(t).strip().lower() for t in v if str(t).strip()}
    except Exception:
        pass
    # Fallback: regex-extract #tags
    return set(re.findall(r"#[a-zA-Z_]+", cell.lower()))


def gran_subset(tags: set, gran: str) -> set:
    return tags & set(TAGS_BY_GRAN[gran])


# ---------------------------------------------------------------------------
# Load + verify
# ---------------------------------------------------------------------------
def load_gt(path: Path) -> dict:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    out = {}
    dropped = 0
    for _, r in df.iterrows():
        key = norm_text(r["Text"])
        if not key:
            dropped += 1
            continue
        tags = (parse_gt_cell(r.get("Statement"))
                | parse_gt_cell(r.get("Phrase"))
                | parse_gt_cell(r.get("Relation")))
        out[key] = tags
    if dropped:
        print(f"  (dropped {dropped} GT rows with empty Text)", file=sys.stderr)
    return out


def load_preds(pred_dir: Path, laws: list, passes: int,
               pattern: str) -> dict:
    """Returns {law -> {pass_idx -> {text_key -> pred_set}}}."""
    out = {law: {} for law in laws}
    for law in laws:
        for k in range(1, passes + 1):
            p = pred_dir / pattern.format(law=law, **{"pass": k})
            if not p.exists():
                print(f"  missing: {p.name}", file=sys.stderr)
                continue
            df = pd.read_csv(p, encoding="utf-8", encoding_errors="replace")
            if "tags" not in df.columns:
                print(f"  no 'tags' column in {p.name}", file=sys.stderr)
                continue
            row_map = {}
            for _, r in df.iterrows():
                row_map[norm_text(r["text"])] = parse_pred_cell(r["tags"])
            out[law][k] = row_map
    return out


def verify_passes_consistent(preds: dict) -> dict:
    """Confirm tags are identical across passes (notebook tags once outside
    the pass loop). Returns a per-law dict with whether tags match."""
    report = {}
    for law, by_pass in preds.items():
        passes = sorted(by_pass.keys())
        if len(passes) <= 1:
            report[law] = "single-pass"
            continue
        ref = by_pass[passes[0]]
        all_eq = True
        for k in passes[1:]:
            other = by_pass[k]
            shared = set(ref) & set(other)
            for t in shared:
                if ref[t] != other[t]:
                    all_eq = False
                    break
            if not all_eq:
                break
        report[law] = "identical" if all_eq else "DIFFER"
    return report


# ---------------------------------------------------------------------------
# Per-row scoring + aggregation
# ---------------------------------------------------------------------------
def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def per_row_metrics(gt: set, pred: set) -> dict:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if not gt else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else (1.0 if not pred else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "exact_match": int(gt == pred),
        "precision": prec,
        "recall": rec,
        "f1": f1(prec, rec),
        "jaccard": (len(gt & pred) / len(gt | pred)) if (gt | pred) else 1.0,
    }


def aggregate(rows: list, gran: str = None) -> dict:
    """Macro + micro across rows. If gran is given, restrict tag sets to
    that granularity bucket on both sides before scoring."""
    n = 0
    em_count = 0
    macro_p = macro_r = macro_f = 0.0
    sum_tp = sum_fp = sum_fn = 0
    for row in rows:
        gt = row["gt"] if gran is None else gran_subset(row["gt"], gran)
        pr = row["pred"] if gran is None else gran_subset(row["pred"], gran)
        m = per_row_metrics(gt, pr)
        n += 1
        em_count += m["exact_match"]
        macro_p += m["precision"]
        macro_r += m["recall"]
        macro_f += m["f1"]
        sum_tp += m["tp"]
        sum_fp += m["fp"]
        sum_fn += m["fn"]
    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
    return {
        "rows": n,
        "exact_match_rate": em_count / n if n else 0.0,
        "macro_precision": macro_p / n if n else 0.0,
        "macro_recall": macro_r / n if n else 0.0,
        "macro_f1": macro_f / n if n else 0.0,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": f1(micro_p, micro_r),
    }


def per_tag_metrics(rows: list) -> list:
    """For each tag in the inventory: support (# rows where GT contains it),
    TP/FP/FN summed over the corpus, P/R/F1."""
    out = []
    for tag in ALL_TAGS:
        tp = fp = fn = support = 0
        for row in rows:
            in_gt = tag in row["gt"]
            in_pr = tag in row["pred"]
            if in_gt:
                support += 1
            if in_gt and in_pr:
                tp += 1
            elif in_gt and not in_pr:
                fn += 1
            elif in_pr and not in_gt:
                fp += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out.append({
            "tag": tag,
            "granularity": TAG_TO_GRAN[tag],
            "support": support,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1(prec, rec), 4),
        })
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    p.add_argument("--pred-pattern",
                   default="dev_pass_{pass}_{law}_step_1_gpt5.csv")
    p.add_argument("--laws", nargs="+", default=DEFAULT_LAWS)
    p.add_argument("--passes", type=int, default=3,
                   help="Used only to verify pass-consistency; "
                        "scoring uses pass 1.")
    p.add_argument("--model-tag", default="gpt5_decomp")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent
                   / "evaluation_output_tags")
    args = p.parse_args()

    out_dir = args.out / args.model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"GT: {args.gt}")
    print(f"pred_dir: {args.pred_dir}")
    print(f"laws: {args.laws}  passes: {args.passes}")
    print(f"out: {out_dir}\n")

    gt_map = load_gt(args.gt)
    print(f"loaded {len(gt_map)} GT rows")

    preds = load_preds(args.pred_dir, args.laws, args.passes, args.pred_pattern)
    consistency = verify_passes_consistent(preds)
    print(f"pass-consistency: {consistency}")
    print("(if all 'identical', pass@3 collapses to pass@1; using pass 1 as "
          "the canonical prediction)\n")

    # Build per-row records using pass 1.
    rows = []
    unmatched = 0
    for law in args.laws:
        if 1 not in preds[law]:
            print(f"!! no pass-1 file for {law}", file=sys.stderr)
            continue
        for text_key, pred_tags in preds[law][1].items():
            if text_key not in gt_map:
                unmatched += 1
                continue
            gt_tags = gt_map[text_key]
            row = {"law": law, "text": text_key, "gt": gt_tags, "pred": pred_tags}
            row.update(per_row_metrics(gt_tags, pred_tags))
            rows.append(row)
    if unmatched:
        print(f"  warning: {unmatched} prediction rows had no GT match",
              file=sys.stderr)
    print(f"matched rows: {len(rows)}")

    # ----- per-row CSV -----
    per_row_path = out_dir / "per_row.csv"
    with open(per_row_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["law", "text", "gt_tags", "pred_tags",
                    "tp", "fp", "fn", "exact_match",
                    "precision", "recall", "f1", "jaccard"])
        for r in rows:
            w.writerow([r["law"], r["text"][:200],
                        " ".join(sorted(r["gt"])),
                        " ".join(sorted(r["pred"])),
                        r["tp"], r["fp"], r["fn"], r["exact_match"],
                        round(r["precision"], 4), round(r["recall"], 4),
                        round(r["f1"], 4), round(r["jaccard"], 4)])
    print(f"wrote {per_row_path}")

    # ----- per-granularity table (compares to IAA from paper §3.2) -----
    iaa_em = {"Statement": 0.758, "Phrase": 0.720, "Relation": 0.577}
    gran_rows = []
    for gran in ["Statement", "Phrase", "Relation"]:
        agg = aggregate(rows, gran=gran)
        agg["granularity"] = gran
        agg["human_iaa_em"] = iaa_em[gran]
        gran_rows.append(agg)
    overall = aggregate(rows)
    overall["granularity"] = "Overall"
    overall["human_iaa_em"] = ""
    gran_rows.append(overall)
    pd.DataFrame(gran_rows)[
        ["granularity", "rows", "exact_match_rate", "human_iaa_em",
         "macro_precision", "macro_recall", "macro_f1",
         "micro_precision", "micro_recall", "micro_f1"]
    ].to_csv(out_dir / "per_granularity.csv", index=False, float_format="%.4f")
    print(f"wrote {out_dir / 'per_granularity.csv'}")

    # ----- per-tag CSV -----
    pd.DataFrame(per_tag_metrics(rows)).to_csv(
        out_dir / "per_tag.csv", index=False)
    print(f"wrote {out_dir / 'per_tag.csv'}")

    # ----- summary -----
    summary = {
        "model_tag": args.model_tag,
        "gt_rows": len(gt_map),
        "matched_rows": len(rows),
        "unmatched_pred_rows": unmatched,
        "pass_consistency": consistency,
        **{f"overall_{k}": v for k, v in overall.items() if k != "granularity"},
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv",
                                   index=False, float_format="%.4f")
    print(f"wrote {out_dir / 'summary.csv'}")

    # ----- console report -----
    print("\n--- Per-granularity (LLM tagging vs human-vs-human IAA) ---")
    print(f"{'Gran':<12} {'EM':>8} {'IAA EM':>8} {'macro F1':>10} "
          f"{'micro F1':>10}")
    for r in gran_rows:
        iaa = (f"{r['human_iaa_em']:.3f}" if isinstance(r['human_iaa_em'], float)
               else "")
        print(f"{r['granularity']:<12} {r['exact_match_rate']:>8.3f} "
              f"{iaa:>8} {r['macro_f1']:>10.3f} {r['micro_f1']:>10.3f}")

    print("\n--- Worst tags by F1 (likely retrieval bottlenecks) ---")
    pt = sorted(per_tag_metrics(rows), key=lambda x: x["f1"])
    print(f"{'tag':<22} {'gran':<10} {'support':>8} {'P':>8} {'R':>8} {'F1':>8}")
    for r in pt[:10]:
        print(f"{r['tag']:<22} {r['granularity']:<10} {r['support']:>8} "
              f"{r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1']:>8.3f}")


if __name__ == "__main__":
    main()
