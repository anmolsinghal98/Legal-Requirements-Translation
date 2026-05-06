"""
Harvest structured elements (Rule, Definition, Exemption, Information, Reference)
from the ground-truth Python code of each law's paragraphs.

Each CSV row is a self-contained paragraph-level script. We exec each one in an
isolated namespace and collect instances of interest. Each harvested instance is
tagged with (state, paragraph_idx, var_name, paragraph_text).
"""
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# code/cross_jurisdictional_demo/harvest.py -> code/cross_jurisdictional_demo -> code
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from class_structure import (
    Section, Expression, Reference, Statement,
    Information, Definition, Rule, Exemption,
)

CORPUS_DIR = HERE / 'corpus'

LAW_FILES = {
    'AR': 'AR-4-110-training.csv',
    'CA': 'CA-1798-training.csv',
    'CT': 'CT-36A-training.csv',
    'MA': 'MA-93H-training.csv',
    'MD': 'MD-14-35-training.csv',
    'MS': 'MS-training.csv',
    'NV': 'NV-603A-training.csv',
    'NY': 'NY-899-training.csv',
    'OR': 'OR-training.csv',
    'UT': 'UT-training.csv',
    'VA': 'VA-182-training.csv',
    'VT': 'VT-training.csv',
    'WI': 'WI-134-training.csv',
}

RULE_TYPE_NAMES = {0: 'OBLIGATION', 1: 'PERMISSION', 2: 'PROHIBITION', 3: 'PENALTY'}


@dataclass
class Item:
    state: str
    para_idx: int
    para_text: str
    var_name: str
    obj: Any


@dataclass
class LawView:
    state: str
    paragraphs: List[str] = field(default_factory=list)
    rules: List[Item] = field(default_factory=list)
    definitions: List[Item] = field(default_factory=list)
    exemptions: List[Item] = field(default_factory=list)
    informations: List[Item] = field(default_factory=list)
    relationship_targets: List[Item] = field(default_factory=list)
    exec_failures: List[int] = field(default_factory=list)


def harvest_law(state: str, csv_path: str) -> LawView:
    view = LawView(state=state)
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for idx, row in enumerate(rows):
        text = (row.get('text') or '').strip()
        code = (row.get('code') or '').strip()
        view.paragraphs.append(text)
        if not code:
            continue
        ns: Dict[str, Any] = {
            'Section': Section, 'Expression': Expression, 'Reference': Reference,
            'Statement': Statement, 'Information': Information, 'Definition': Definition,
            'Rule': Rule, 'Exemption': Exemption,
        }
        try:
            exec(code, ns)
        except Exception:
            view.exec_failures.append(idx)
            continue
        for name, val in ns.items():
            if name.startswith('__') or name in ns and not isinstance(val, (Rule, Definition, Exemption, Information, Reference)):
                continue
            item = Item(state=state, para_idx=idx, para_text=text, var_name=name, obj=val)
            if isinstance(val, Rule):
                view.rules.append(item)
            elif isinstance(val, Definition):
                view.definitions.append(item)
            elif isinstance(val, Exemption):
                view.exemptions.append(item)
            elif isinstance(val, Information):
                view.informations.append(item)
            elif isinstance(val, Reference):
                pass  # top-level References (rare); we'll harvest from relationships instead
        # Also harvest all relationship-target Expressions (these are the de-facto "references")
        for group in (view.rules, view.definitions, view.exemptions, view.informations):
            for it in group:
                if it.para_idx != idx:
                    continue
                for rel_name, targets in it.obj.relationships.items():
                    for t in targets:
                        if isinstance(t, (Expression,)):
                            view.relationship_targets.append(Item(
                                state=state, para_idx=idx, para_text=text,
                                var_name=f"{it.var_name}.{rel_name}",
                                obj=t,
                            ))
    return view


def load_all_laws() -> Dict[str, LawView]:
    return {state: harvest_law(state, str(CORPUS_DIR / fname))
            for state, fname in LAW_FILES.items()}


def _expr_text(e) -> str:
    if e is None:
        return ''
    if isinstance(e, list):
        return ' '.join(_expr_text(x) for x in e)
    if hasattr(e, 'text'):
        return e.text or ''
    return str(e)


def summarize(laws: Dict[str, LawView]) -> None:
    print(f"{'State':<6} {'Paras':>6} {'Rules':>6} {'Defs':>5} {'Exempt':>7} {'Info':>5} {'RelTgt':>7} {'Fails':>6}")
    print('-' * 60)
    total = {'p': 0, 'r': 0, 'd': 0, 'x': 0, 'i': 0, 'rt': 0, 'f': 0}
    for state, view in laws.items():
        print(f"{state:<6} {len(view.paragraphs):>6} {len(view.rules):>6} {len(view.definitions):>5} "
              f"{len(view.exemptions):>7} {len(view.informations):>5} {len(view.relationship_targets):>7} "
              f"{len(view.exec_failures):>6}")
        total['p'] += len(view.paragraphs)
        total['r'] += len(view.rules)
        total['d'] += len(view.definitions)
        total['x'] += len(view.exemptions)
        total['i'] += len(view.informations)
        total['rt'] += len(view.relationship_targets)
        total['f'] += len(view.exec_failures)
    print('-' * 60)
    print(f"{'TOTAL':<6} {total['p']:>6} {total['r']:>6} {total['d']:>5} {total['x']:>7} "
          f"{total['i']:>5} {total['rt']:>7} {total['f']:>6}")


if __name__ == '__main__':
    laws = load_all_laws()
    summarize(laws)
