"""
Six cross-jurisdictional queries over the harvested ground-truth representations.
Each query returns a dict {state -> list of hits}; each hit is a small summary record.
"""
import re
from collections import defaultdict
from typing import Dict, List

from harvest import LawView, Item, load_all_laws, RULE_TYPE_NAMES, _expr_text


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s or '').strip()


def _rule_summary(it: Item) -> Dict:
    r = it.obj
    entity = _expr_text(r.entity)
    description = _expr_text(r.description)
    conditions = [_expr_text(c) for c in r.conditions]
    return {
        'state': it.state,
        'para': it.para_idx,
        'type': RULE_TYPE_NAMES.get(r.rule_type, 'UNKNOWN'),
        'entity': _norm(entity),
        'description': _norm(description),
        'conditions': [_norm(c) for c in conditions if c],
        'has_exception': [_norm(_expr_text(x)) for x in r.relationships.get('has_exception', [])],
        'is_exception_to': [_norm(_expr_text(x)) for x in r.relationships.get('is_exception_to', [])],
        'follows': [_norm(_expr_text(x)) for x in r.relationships.get('follows', [])],
    }


# -----------------------------------------------------------------------------
# Q1: Rules that require notifying the Attorney General.
# -----------------------------------------------------------------------------
def q1_attorney_general(laws: Dict[str, LawView]) -> Dict[str, List[Dict]]:
    pat = re.compile(r'\battorney\s+general\b', re.IGNORECASE)
    out: Dict[str, List[Dict]] = defaultdict(list)
    for state, view in laws.items():
        for it in view.rules:
            if it.obj.rule_type != 0:  # OBLIGATION only
                continue
            text = _expr_text(it.obj.description) + ' ' + _expr_text(it.obj.entity)
            if pat.search(text):
                out[state].append(_rule_summary(it))
    return out


# -----------------------------------------------------------------------------
# Q2: Definitions of "personal information" (with exclusions).
# -----------------------------------------------------------------------------
def q2_personal_information(laws: Dict[str, LawView]) -> Dict[str, List[Dict]]:
    pat = re.compile(r'personal\s+information', re.IGNORECASE)
    out: Dict[str, List[Dict]] = defaultdict(list)
    for state, view in laws.items():
        for it in view.definitions:
            term = _expr_text(it.obj.defined_term)
            if not pat.search(term):
                continue
            out[state].append({
                'state': state,
                'para': it.para_idx,
                'term': _norm(term),
                'meaning': [_norm(_expr_text(m)) for m in it.obj.meaning],
                'exclusions': [_norm(_expr_text(x)) for x in it.obj.exclusions],
            })
    return out


# -----------------------------------------------------------------------------
# Q3: Rules that permit delaying notification due to law enforcement.
# -----------------------------------------------------------------------------
def q3_law_enforcement_delay(laws: Dict[str, LawView]) -> Dict[str, List[Dict]]:
    law_enforcement = re.compile(r'law\s+enforcement', re.IGNORECASE)
    delay = re.compile(r'\b(delay|delayed|postpon)', re.IGNORECASE)
    out: Dict[str, List[Dict]] = defaultdict(list)
    for state, view in laws.items():
        for it in view.rules:
            if it.obj.rule_type != 1:  # PERMISSION
                continue
            desc = _expr_text(it.obj.description)
            cond_texts = [_expr_text(c) for c in it.obj.conditions]
            cond_blob = ' '.join(cond_texts)
            # The rule is a delay permission if "delay" appears in its description
            # and "law enforcement" appears anywhere in desc or conditions.
            if delay.search(desc) and law_enforcement.search(desc + ' ' + cond_blob):
                out[state].append({
                    'state': state,
                    'para': it.para_idx,
                    'description': _norm(desc),
                    'conditions': [_norm(c) for c in cond_texts if c],
                })
    return out


# -----------------------------------------------------------------------------
# Q4: Cross-references to federal statutes (HIPAA, GLBA, FCRA, COPPA, FERPA).
# -----------------------------------------------------------------------------
FEDERAL_STATUTES = [
    (r'\bHIPAA\b|Health\s+Insurance\s+Portability\s+and\s+Accountability\s+Act', 'HIPAA'),
    (r'\bGLBA\b|Gramm[\-\s]Leach[\-\s]Bliley', 'GLBA'),
    (r'\bFCRA\b|Fair\s+Credit\s+Reporting\s+Act', 'FCRA'),
    (r"\bCOPPA\b|Children'?s\s+Online\s+Privacy\s+Protection", 'COPPA'),
    (r'\bFERPA\b|Family\s+Educational\s+Rights\s+and\s+Privacy', 'FERPA'),
    (r'Title\s+V\s+of\s+the\s+Gramm|15\s+U\.?S\.?C\.?\s+6801', 'GLBA'),
]


def q4_federal_references(laws: Dict[str, LawView]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    compiled = [(re.compile(p, re.IGNORECASE), label) for p, label in FEDERAL_STATUTES]
    for state, view in laws.items():
        seen = set()
        for it in view.relationship_targets + [
            Item(state=state, para_idx=0, para_text='', var_name='',
                 obj=type('X', (), {'text': _expr_text(x.obj.entity) + ' ' + _expr_text(x.obj.description)})())
            for x in view.rules
        ]:
            text = getattr(it.obj, 'text', '')
            if not text:
                continue
            for pat, label in compiled:
                if pat.search(text):
                    key = (it.state, label, it.para_idx)
                    if key in seen:
                        continue
                    seen.add(key)
                    out[state].append({
                        'state': state,
                        'para': it.para_idx,
                        'federal_statute': label,
                        'excerpt': _norm(text)[:160],
                    })
        # also check paragraph text (ground truth code may omit the federal label in a Reference object)
        for idx, text in enumerate(view.paragraphs):
            for pat, label in compiled:
                if pat.search(text):
                    key = (state, label, idx)
                    if key in seen:
                        continue
                    seen.add(key)
                    out[state].append({
                        'state': state,
                        'para': idx,
                        'federal_statute': label,
                        'excerpt': _norm(text)[:160],
                    })
    return out


# -----------------------------------------------------------------------------
# Q5: Penalty amounts for data-breach violations.
# -----------------------------------------------------------------------------
DOLLAR = re.compile(r'(\$[\d,]+(?:\.\d+)?|\b\d{1,3}(?:,\d{3})*\s*dollars?|\b(?:one|two|three|four|five|ten|twenty|fifty|one\s+hundred|five\s+hundred|one\s+thousand|five\s+thousand|ten\s+thousand|fifty\s+thousand|one\s+hundred\s+thousand|one\s+million)\s+(?:thousand\s+)?dollars?)', re.IGNORECASE)


def q5_penalties(laws: Dict[str, LawView]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for state, view in laws.items():
        for it in view.rules:
            if it.obj.rule_type != 3:  # PENALTY
                continue
            desc = _expr_text(it.obj.description)
            if not desc:
                continue
            amounts = DOLLAR.findall(desc)
            # flatten any tuple capture
            amounts_flat = []
            for a in amounts:
                if isinstance(a, tuple):
                    amounts_flat.append(next((x for x in a if x), ''))
                else:
                    amounts_flat.append(a)
            out[state].append({
                'state': state,
                'para': it.para_idx,
                'description': _norm(desc)[:180],
                'amounts_extracted': [a.strip() for a in amounts_flat if a.strip()],
            })
    return out


# -----------------------------------------------------------------------------
# Q6: Rules with explicit exceptions (non-empty has_exception or is_exception_to).
# -----------------------------------------------------------------------------
def q6_exceptions(laws: Dict[str, LawView]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for state, view in laws.items():
        for it in view.rules:
            rels = it.obj.relationships
            has_exc = rels.get('has_exception', [])
            is_exc_to = rels.get('is_exception_to', [])
            if not has_exc and not is_exc_to:
                continue
            out[state].append({
                'state': state,
                'para': it.para_idx,
                'rule_type': RULE_TYPE_NAMES.get(it.obj.rule_type, '?'),
                'rule_description': _norm(_expr_text(it.obj.description))[:120],
                'exceptions': [_norm(_expr_text(x))[:120] for x in has_exc],
                'exception_to': [_norm(_expr_text(x))[:120] for x in is_exc_to],
            })
    return out


def summarize_counts(query_name: str, results: Dict[str, List[Dict]]) -> None:
    total = sum(len(v) for v in results.values())
    hit_states = sum(1 for v in results.values() if v)
    print(f"{query_name:40s}  hits={total:4d}  states={hit_states}/13")


if __name__ == '__main__':
    laws = load_all_laws()
    q1 = q1_attorney_general(laws)
    q2 = q2_personal_information(laws)
    q3 = q3_law_enforcement_delay(laws)
    q4 = q4_federal_references(laws)
    q5 = q5_penalties(laws)
    q6 = q6_exceptions(laws)

    print()
    summarize_counts('Q1 rules requiring AG notification', q1)
    summarize_counts('Q2 definitions of personal information', q2)
    summarize_counts('Q3 law-enforcement delay permissions', q3)
    summarize_counts('Q4 cross-references to federal statutes', q4)
    summarize_counts('Q5 penalty rules with amounts', q5)
    summarize_counts('Q6 rules with explicit exceptions', q6)

    # Sample Q3 output (the side-by-side query)
    print('\n--- Q3 detail ---')
    for state, hits in q3.items():
        for h in hits:
            print(f"{state} (p{h['para']}): {h['description'][:60]}  conditions={len(h['conditions'])}")
