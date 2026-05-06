"""
Evaluation library: compilation, structural (5 tests), semantic (16 tests),
and pass@k. Lifted from these published notebooks so model outputs are scored
under the exact same logic as the RE 2025 paper:

  code/Code-Gen-Compliation-Testing.ipynb
  code/Code-Gen-Structural-Testing.ipynb
  code/Code-Gen-Semantic-Testing.ipynb
  code/Compute-pass-at-k.ipynb
"""

import ast
import contextlib
import io
import sys
import unittest
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Make the published class_structure / serialize / test_statements importable
# so we use the exact functions cited in the paper rather than reimplementing.
# This file lives at code/evaluation/eval_lib.py; the modules sit in code/.
# ---------------------------------------------------------------------------
LRT_CODE = Path(__file__).resolve().parent.parent
if str(LRT_CODE) not in sys.path:
    sys.path.insert(0, str(LRT_CODE))

from class_structure import (  # noqa: E402
    Section, Expression, Statement, Information, Definition, Rule,
    Exemption, Reference,
)
from serialize import serialize_section  # noqa: E402
from test_statements import (  # noqa: E402
    test_information_description,
    test_definition_term,
    test_definition_meaning,
    test_definition_exclusions,
    test_rule_entity,
    test_rule_type,
    test_rule_description,
    test_rule_conditions,
    test_exemption_description,
    test_reference_relationship,
    test_statement_relationship,
)

TOTAL_STRUCTURAL_TESTS = 5
TOTAL_SEMANTIC_TESTS = 16


# ---------------------------------------------------------------------------
# Code preprocessing + safe execution
# ---------------------------------------------------------------------------
def preprocess_code(code) -> str:
    """Strip ```python … ``` fences and any trailing chatter."""
    if not isinstance(code, str):
        return ""
    code = code.strip()
    flag = 0
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
        flag = 1
    elif code.startswith("```"):
        code = code[3:].strip()
        flag = 1
    if code.endswith("```"):
        code = code[:-3].strip()
    elif flag == 1:
        code = code.split("```")[0].strip()
    return code


def is_valid_python_code(code: str) -> bool:
    """Compile + exec in an isolated namespace. Returns True on success.
    Stdout/stderr are silenced so test rows that print don't pollute logs."""
    try:
        ns = {}
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            obj = compile(code, "<gen>", "exec")
            exec(obj, globals(), ns)
        return True
    except (SyntaxError, Exception):
        return False


def run_code_string(code_str: str) -> List[Section]:
    """Same as the notebook helper. Returns top-level Section instances."""
    try:
        ns = {}
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            exec(code_str, globals(), ns)
        return [v for v in ns.values() if isinstance(v, Section)]
    except (SyntaxError, Exception):
        return []


# ---------------------------------------------------------------------------
# String / list comparison with edit-distance threshold (from structural NB)
# ---------------------------------------------------------------------------
def compare_strings_with_threshold(s1, s2, threshold: int = 10):
    s1 = "" if s1 is None else str(s1)
    s2 = "" if s2 is None else str(s2)
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return (dp[m][n] <= threshold, dp[m][n])


def compare_list_strings_with_threshold(list1, list2, threshold: int = 10) -> bool:
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        ok, _ = compare_strings_with_threshold(a, b, threshold)
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# Structural tests (5)
# ---------------------------------------------------------------------------
def _serialize_top(code: str):
    return [serialize_section(s) for s in run_code_string(code)]


def _struct_eq_len(d1, d2) -> bool:
    return len(d1) == len(d2)


def test_section_number(gt_code: str, gen_code: str) -> int:
    try:
        d1, d2 = _serialize_top(gt_code), _serialize_top(gen_code)
        if not _struct_eq_len(d1, d2):
            return 0
        return int(compare_list_strings_with_threshold(
            [d["sectionNumber"] for d in d1],
            [d["sectionNumber"] for d in d2], threshold=10))
    except Exception:
        return 0


def test_section_name(gt_code: str, gen_code: str) -> int:
    try:
        d1, d2 = _serialize_top(gt_code), _serialize_top(gen_code)
        if not _struct_eq_len(d1, d2):
            return 0
        return int(compare_list_strings_with_threshold(
            [d["sectionTitle"] for d in d1],
            [d["sectionTitle"] for d in d2], threshold=10))
    except Exception:
        return 0


def test_section_subsections(gt_code: str, gen_code: str) -> int:
    try:
        d1, d2 = _serialize_top(gt_code), _serialize_top(gen_code)
        if not _struct_eq_len(d1, d2):
            return 0
        for a, b in zip(d1, d2):
            if len(a["subSections"]) != len(b["subSections"]):
                return 0
        return 1
    except Exception:
        return 0


def test_section_expressions(gt_code: str, gen_code: str) -> int:
    try:
        d1, d2 = _serialize_top(gt_code), _serialize_top(gen_code)
        if not _struct_eq_len(d1, d2):
            return 0
        for a, b in zip(d1, d2):
            if len(a["expressions"]) != len(b["expressions"]):
                return 0
        return 1
    except Exception:
        return 0


def test_section_statements(gt_code: str, gen_code: str) -> int:
    try:
        d1, d2 = _serialize_top(gt_code), _serialize_top(gen_code)
        if not _struct_eq_len(d1, d2):
            return 0
        for a, b in zip(d1, d2):
            if len(a["statements"]) != len(b["statements"]):
                return 0
        return 1
    except Exception:
        return 0


def structural_test_row(gt_code: str, gen_code: str) -> dict:
    return {
        "section_number": test_section_number(gt_code, gen_code),
        "section_name": test_section_name(gt_code, gen_code),
        "section_subsections": test_section_subsections(gt_code, gen_code),
        "section_expressions": test_section_expressions(gt_code, gen_code),
        "section_statements": test_section_statements(gt_code, gen_code),
    }


# ---------------------------------------------------------------------------
# Semantic tests (16). Mirrors create_result_df() from the semantic notebook.
# Each per-attribute cell stores the raw (passed, validity) tuple so the
# pass@k distribution computation can reparse them later.
# ---------------------------------------------------------------------------
SEM_FIELDS = [
    ("Information Description", lambda tc, gt, gn: test_information_description(tc, gt, gn)),
    ("Definition Term",          lambda tc, gt, gn: test_definition_term(tc, gt, gn)),
    ("Definition Meaning",       lambda tc, gt, gn: test_definition_meaning(tc, gt, gn)),
    ("Definition Exclusions",    lambda tc, gt, gn: test_definition_exclusions(tc, gt, gn)),
    ("Rule Entity",              lambda tc, gt, gn: test_rule_entity(tc, gt, gn)),
    ("Rule Type",                lambda tc, gt, gn: test_rule_type(tc, gt, gn)),
    ("Rule Description",         lambda tc, gt, gn: test_rule_description(tc, gt, gn)),
    ("Rule Conditions",          lambda tc, gt, gn: test_rule_conditions(tc, gt, gn)),
    ("Exemption Description",    lambda tc, gt, gn: test_exemption_description(tc, gt, gn)),
    ("Refines",                  lambda tc, gt, gn: test_statement_relationship(tc, gt, gn, relation="refines")),
    ("Is Refined By",            lambda tc, gt, gn: test_statement_relationship(tc, gt, gn, relation="is_refined_by")),
    ("Follows",                  lambda tc, gt, gn: test_statement_relationship(tc, gt, gn, relation="follows")),
    ("Is Followed By",           lambda tc, gt, gn: test_statement_relationship(tc, gt, gn, relation="is_followed_by")),
    ("Exceptions",               lambda tc, gt, gn: test_statement_relationship(tc, gt, gn, relation="has_exception")),
    ("Is Exception To",          lambda tc, gt, gn: test_statement_relationship(tc, gt, gn, relation="is_exception_to")),
    ("References",               lambda tc, gt, gn: test_reference_relationship(tc, gt, gn)),
]


def _safe_call(fn, tc, gt, gn):
    """Each test_statements function may exec untrusted code. Catch anything."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return fn(tc, gt, gn)
    except Exception:
        return (0, 0)


def semantic_test_row(gt_code: str, gen_code: str) -> dict:
    """Returns a dict matching the columns produced by the notebook's
    create_result_df() — minus 'text', 'GT Code', 'Generated Code' which the
    caller fills in. Per-attribute cells store the raw (p, v) tuple."""
    tc = unittest.TestCase()
    total_passed, tp, fp, fn_count, mismatch = 0, 0, 0, 0, 0
    test_map = {}
    for label, fn_call in SEM_FIELDS:
        p, v = _safe_call(fn_call, tc, gt_code, gen_code)
        test_map[label] = (p, v)
        total_passed += int(p)
        if not p and v == 1:
            fn_count += 1
        elif not p and v == 2:
            mismatch += 1
        elif not p and not v:
            fp += 1
        elif p and v:
            tp += 1
    row_acc = total_passed / TOTAL_SEMANTIC_TESTS
    row_rec = tp / (tp + fn_count + mismatch) if (tp + fn_count + mismatch) else None
    row_pre = tp / (tp + fp + mismatch) if (tp + fp + mismatch) else None
    out = {
        "Total Passed": total_passed,
        "Mismacthes": mismatch,  # spelling matches the notebook
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn_count,
        "Accuracy": row_acc,
        "Recall": row_rec,
        "Precision": row_pre,
    }
    out.update(test_map)
    return out


# ---------------------------------------------------------------------------
# IO helpers: normalize generated dataframes to a (text, code) shape
# ---------------------------------------------------------------------------
GENERATED_CODE_COLS = ["code", "code step 3", "code step 2", "code step 1"]


def normalize_gen_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the first available code column, preprocess fences, return df with
    columns ['text', 'code']."""
    code_col = next((c for c in GENERATED_CODE_COLS if c in df.columns), None)
    if code_col is None:
        raise ValueError(f"no code column found in {list(df.columns)}")
    out = pd.DataFrame({
        "text": df["text"].astype(str),
        "code": df[code_col].apply(preprocess_code),
    })
    return out


def normalize_gt_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "text": df["text"].astype(str),
        "code": df["code"].apply(preprocess_code),
    })
    return out


def align_rows(gt_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    """Truncate to the shorter length so a partial run still scores cleanly."""
    n = min(len(gt_df), len(gen_df))
    if len(gt_df) != len(gen_df):
        print(f"  WARN: gt has {len(gt_df)} rows, gen has {len(gen_df)} — "
              f"using first {n}", file=sys.stderr)
    return gt_df.iloc[:n].reset_index(drop=True), gen_df.iloc[:n].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-(law, pass) builders
# ---------------------------------------------------------------------------
def build_compilation_df(gen_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, r in gen_df.iterrows():
        ok = is_valid_python_code(r["code"]) if r["code"] else False
        rows.append({"text": r["text"], "code": r["code"], "compiles": int(ok)})
    return pd.DataFrame(rows)


def build_structural_df(gt_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, gt in gt_df.iterrows():
        gn = gen_df.iloc[i]
        m = structural_test_row(gt["code"], gn["code"])
        m["Total Passed"] = sum(m.values())
        m["text"] = gt["text"]
        m["GT Code"] = gt["code"]
        m["Generated Code"] = gn["code"]
        rows.append(m)
    cols_front = ["text", "GT Code", "Generated Code", "Total Passed"]
    df = pd.DataFrame(rows)
    return df[cols_front + [c for c in df.columns if c not in cols_front]]


def build_semantic_df(gt_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, gt in gt_df.iterrows():
        gn = gen_df.iloc[i]
        m = semantic_test_row(gt["code"], gn["code"])
        m["text"] = gt["text"]
        m["GT Code"] = gt["code"]
        m["Generated Code"] = gn["code"]
        rows.append(m)
    df = pd.DataFrame(rows)
    cols_front = [
        "text", "GT Code", "Generated Code",
        "Total Passed", "Mismacthes", "True Positives",
        "False Positives", "False Negatives",
        "Accuracy", "Recall", "Precision",
    ]
    return df[cols_front + [c for c in df.columns if c not in cols_front]]


# ---------------------------------------------------------------------------
# Pass@k over a list of semantic dataframes
# ---------------------------------------------------------------------------
def consolidate_pass_at_k(df_list: list) -> pd.DataFrame:
    """Per row, pick the dataframe whose 'Total Passed' is max (first wins)."""
    rows = len(df_list[0])
    consolidated = []
    for i in range(rows):
        best, best_idx = -1, 0
        for j, df in enumerate(df_list):
            v = df["Total Passed"].iloc[i]
            if v > best:
                best, best_idx = v, j
        consolidated.append(df_list[best_idx].iloc[i])
    return pd.DataFrame(consolidated).reset_index(drop=True)


def _parse_pv_cell(cell):
    """Re-parse a (p, v) cell that may have been written and re-read as a
    string '(1, 1)'. Returns (p, v) ints, or None on failure."""
    if isinstance(cell, tuple) and len(cell) == 2:
        return int(cell[0]), int(cell[1])
    if isinstance(cell, str):
        try:
            v = ast.literal_eval(cell)
            if isinstance(v, tuple) and len(v) == 2:
                return int(v[0]), int(v[1])
        except Exception:
            pass
    return None


def calculate_distribution(df: pd.DataFrame) -> dict:
    """Per-attribute precision/recall/accuracy, matching the published
    Compute-pass-at-k notebook (read raw (p, v) tuples)."""
    skip = {"text", "GT Code", "Generated Code", "Total Passed",
            "True Positives", "False Positives", "False Negatives",
            "Accuracy", "Recall", "Precision", "Mismacthes"}
    dist = {}
    n = len(df)
    for col in df.columns:
        if col in skip:
            continue
        tp = fp = fn = passed = 0
        for cell in df[col]:
            pv = _parse_pv_cell(cell)
            if pv is None:
                continue
            p, v = pv
            if p == 1:
                passed += 1
            if p == 1 and v == 1:
                tp += 1
            elif p == 0 and (v == 0 or v == 2):
                fp += 1
            elif p == 0 and (v == 1 or v == 2):
                fn += 1
        acc = passed / n if n else None
        rec = tp / (tp + fn) if (tp + fn) else None
        pre = tp / (tp + fp) if (tp + fp) else None
        dist[col] = {"Accuracy": acc, "Recall": rec, "Precision": pre}
    return dist


def metrics_from_consolidated(df: pd.DataFrame) -> dict:
    """Mean accuracy/recall/precision over rows + pass@k count."""
    n = len(df)
    nn_acc = df["Accuracy"].notna().sum()
    nn_rec = df["Recall"].notna().sum()
    nn_pre = df["Precision"].notna().sum()
    return {
        "rows": int(n),
        "mean_accuracy": float(df["Accuracy"].sum() / nn_acc) if nn_acc else 0.0,
        "mean_recall": float(df["Recall"].sum() / nn_rec) if nn_rec else 0.0,
        "mean_precision": float(df["Precision"].sum() / nn_pre) if nn_pre else 0.0,
        "all_pass_count": int((df["Total Passed"] == TOTAL_SEMANTIC_TESTS).sum()),
        "pass_at_k": float((df["Total Passed"] == TOTAL_SEMANTIC_TESTS).sum() / n)
                     if n else 0.0,
    }
