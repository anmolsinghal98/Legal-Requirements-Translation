"""
Decomposition pipeline (3-step) on a Together-hosted model.
Mirrors code/Code-with-Decomposition.ipynb:

  1. LLM-tag every test segment using the predefined_tags definitions.
  2. Embed dev set + test set with a Together embedding model.
  3. Demo selection: tag-overlap then cosine similarity; shuffle, take top-10,
     random-sample 3.
  4. Step 1 prompt -> code step 1   (minimal class structure)
  5. Step 2 prompt -> code step 2   (adds Information/Definition/Rule/Exemption)
  6. Step 3 prompt -> code step 3   (adds Reference + relationships)

Each pass writes three CSVs under --output-dir:
  <law>-<model>-step1-<pass>.csv
  <law>-<model>-step2-<pass>.csv
  <law>-<model>-step3-<pass>.csv

Run from code/. Usage:
  python run_qwen_decomposition.py \
      --test "../test files/COPPA.csv" \
      --output-dir qwen_results/decomp --law-tag COPPA \
      --passes 3 --dry-run 3
"""

import argparse
import ast
import csv
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from together import Together
from together.error import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    APIError,
)


# ---------------------------------------------------------------------------
# Class-structure strings (verbatim from Code-with-Decomposition.ipynb)
# ---------------------------------------------------------------------------
CODE_STRING_STEP1 = """
from typing import List, Optional

class Section:
    \"""
    A bullet point in the legal text. Every bullet point starts a new Section,
    and sub-bullet points become subSections.

    Attributes:
        sectionNumber (str): The identifying number or label of this Section.
        sectionTitle (str): An optional title for this Section.
        parent (Optional[Section]): The parent Section if this is a nested (sub-)Section,
            otherwise None for a top-level Section.
        subSections (List[Section]): Any child Sections nested under this Section.
        expressions (List[Expression]): The Expression objects contained directly in this Section.
        statements (List[Statement]): The Statement objects contained directly in this Section.

    Methods:
        add_subsection(subsection: 'Section'):
            Adds a subsection (child) to this Section and sets the subsection's parent to self.

        add_expression(expression: 'Expression'):
            Adds an Expression object to this Section's expressions list.

        add_statement(statement: 'Statement'):
            Adds a Statement object to this Section's statements list.
    \"""

    def __init__(self,sectionNumber: str, sectionTitle: str = "", parent=None):
        self.sectionNumber: str = sectionNumber
        self.sectionTitle: str = sectionTitle
        self.parent: Optional['Section'] = parent
        self.subSections: List['Section'] = []
        self.expressions: List['Expression'] = []
        self.statements: List['Statement'] = []

    def add_subsection(self, subsection: 'Section'):
        self.subSections.append(subsection)
        subsection.parent = self
    def add_expression(self, expression: 'Expression'):
        self.expressions.append(expression)
    def add_statement(self, statement: 'Statement'):
        self.statements.append(statement)


class Expression:
    \"""
    A snippet of text within one bullet point (Section). Represents the smallest
    textual unit that can contain references or other embedded elements.

    Each Expression belongs to exactly one Section.

    Attributes:
        section (Section): The Section in which this Expression is found.
        text (str): The textual content of the Expression.
        includes (Optional[List[Expression]]): A child Expression in a subsection that this Expression includes
    \"""

    def __init__(self, section: Section, text: str, includes=None):
        self.section: Section = section
        section.add_expression(self)
        self.text: str = text
        self.includes: Optional[List[Expression]] = includes if includes is not None else []


class Statement:
    \"""
    A legal statement that can span multiple bullet points (Sections) if those
    bullet points are nested under a single conceptual clause. Statements often
    contain or refer to multiple Expressions.

    Attributes:
        section (Section): The Section that represents
            the location in the text where this Statement starts.
        text (str): The full textual content of the Statement.
    \"""

    def __init__(self, section: Optional[Section] = None, text: str = ""):
        self.sections: Section = section
        self.text: str = text
"""


CODE_STRING_STEP2 = """
class Section:
    \"""(same as step 1)\"""
    def __init__(self,sectionNumber: str, sectionTitle: str = "", parent=None):
        self.sectionNumber: str = sectionNumber
        self.sectionTitle: str = sectionTitle
        self.parent: Optional['Section'] = parent
        self.subSections: List['Section'] = []
        self.expressions: List['Expression'] = []
        self.statements: List['Statement'] = []

    def add_subsection(self, subsection: 'Section'):
        self.subSections.append(subsection)
        subsection.parent = self
    def add_expression(self, expression: 'Expression'):
        self.expressions.append(expression)
    def add_statement(self, statement: 'Statement'):
        self.statements.append(statement)


class Expression:
    \"""(same as step 1)\"""
    def __init__(self, section: Section, text: str, includes=None):
        self.section: Section = section
        section.add_expression(self)
        self.text: str = text
        self.includes: Optional[List[Expression]] = includes if includes is not None else []


class Statement:
    \"""(same as step 1)\"""
    def __init__(self, section: Optional[Section] = None, text: str = ""):
        self.sections: Section = section
        self.text: str = text


class Information(Statement):
    \"""
    A type of Statement that represents something that is known or proved to be true.

    Attributes:
        description (List[Expression]): The Expressions that contains the factual information.
    \"""
    def __init__(self, section, description: Expression):
        super().__init__(section)
        self.description: List[Expression] = []
        if description is not None:
            self.description.append(description)


class Definition(Statement):
    \"""
    A type of Statement that defines a concept or term in the legal text.

    Attributes:
        defined_term (Expression): The Expression stating the term being defined.
        meaning (List[Expression]): One or more Expressions elaborating the meaning of the term.
        exclusions (List[Expression]): Expressions clarifying what the term excludes or does not cover.
    \"""
    def __init__(self, section, defined_term: Expression):
        super().__init__(section)
        self.defined_term: Expression = defined_term
        self.meaning: List[Expression] = []
        self.exclusions: List[Expression] = []


class Rule(Statement):
    \"""
    A Statement describing a legal rule, which may take one of four types: obligation,
    permission, prohibition, or penalty.

    Attributes:
        rule_type (int): One of OBLIGATION, PERMISSION, PROHIBITION, PENALTY.
        entity (Expression): The main entity to which the rule applies.
        description (Expression): An Expression describing the rule.
        conditions (List[Expression]): Conditions under which the rule applies [if, when, after].
    \"""
    OBLIGATION = 0
    PERMISSION = 1
    PROHIBITION = 2
    PENALTY = 3

    def __init__(self, section, entity: Expression):
        super().__init__(section)
        self.rule_type: int = None
        self.entity: Expression = entity
        self.description: Optional[Expression] = None
        self.conditions: List[Expression] = []


class Exemption(Statement):
    \"""
    A type of Statement indicating that a person, object, or situation is exempt
    from another rule or requirement.

    Attributes:
        description (List[Expression]): One or more Expressions describing the exemption.
    \"""
    def __init__(self, section=None, description: Optional[Expression] = None):
        super().__init__(section)
        self.description: List[Expression] = []
        if description is not None:
            self.description.append(description)
"""


CODE_STRING_STEP3 = """
class Section:
    \"""(same as previous steps)\"""
    def __init__(self,sectionNumber: str, sectionTitle: str = "", parent=None):
        self.sectionNumber: str = sectionNumber
        self.sectionTitle: str = sectionTitle
        self.parent: Optional['Section'] = parent
        self.subSections: List['Section'] = []
        self.expressions: List['Expression'] = []
        self.statements: List['Statement'] = []
    def add_subsection(self, subsection: 'Section'):
        self.subSections.append(subsection); subsection.parent = self
    def add_expression(self, expression: 'Expression'):
        self.expressions.append(expression)
    def add_statement(self, statement: 'Statement'):
        self.statements.append(statement)


class Expression:
    \"""(same as previous steps)\"""
    def __init__(self, section: Section, text: str, includes=None):
        self.section: Section = section
        section.add_expression(self)
        self.text: str = text
        self.includes: Optional[List[Expression]] = includes if includes is not None else []


class Reference(Expression):
    \"""
    A type of Expression that refers to another part of the legal text, including pointers,
    numbers, or names to other sections, paragraphs, or laws.

    Attributes:
        target (Union[Expression, Statement]): The target this Reference points to.
    \"""
    def __init__(self, section: Section, text: str, target: 'Statement'):
        super().__init__(section, text)
        self.target: 'Statement' = target


class Statement:
    \"""
    A legal statement spanning one or more Sections. Replaces the step-2 `text`
    attribute with a relationships dict.

    Relationship keys:
        - "refines": References/Statements this Statement refines.
        - "is_refined_by": References/Statements that refine this Statement (as defined in, as described in).
        - "has_exception": References/Statements that are exceptions (unless, except).
        - "is_exception_to": References/Statements for which this is an exception (notwithstanding).
        - "follows": Pre-conditions (pursuant to, in accordance with, under).
        - "is_followed_by": References/Statements that follow this Statement.
    \"""
    def __init__(self, section: Optional[Section] = None):
        self.sections: Section = section
        self.relationships = {
            "refines": [], "is_refined_by": [],
            "has_exception": [], "is_exception_to": [],
            "follows": [], "is_followed_by": []
        }
    def add_refines(self, target): self.relationships["refines"].append(target)
    def add_exception(self, e): self.relationships["has_exception"].append(e)
    def add_follows(self, target): self.relationships["follows"].append(target)
    def add_is_refined_by(self, target): self.relationships["is_refined_by"].append(target)
    def add_is_exception_to(self, e): self.relationships["is_exception_to"].append(e)
    def add_is_followed_by(self, target): self.relationships["is_followed_by"].append(target)


class Information(Statement):
    def __init__(self, section, description: Expression):
        super().__init__(section)
        self.description: List[Expression] = []
        if description is not None:
            self.description.append(description)


class Definition(Statement):
    def __init__(self, section, defined_term: Expression):
        super().__init__(section)
        self.defined_term: Expression = defined_term
        self.meaning: List[Expression] = []
        self.exclusions: List[Expression] = []


class Rule(Statement):
    OBLIGATION = 0
    PERMISSION = 1
    PROHIBITION = 2
    PENALTY = 3
    def __init__(self, section, entity: Expression):
        super().__init__(section)
        self.rule_type: int = None
        self.entity: Expression = entity
        self.description: Optional[Expression] = None
        self.conditions: List[Expression] = []


class Exemption(Statement):
    def __init__(self, section=None, description: Optional[Expression] = None):
        super().__init__(section)
        self.description: List[Expression] = []
        if description is not None:
            self.description.append(description)
"""


# ---------------------------------------------------------------------------
# Prompts (verbatim from the notebook)
# ---------------------------------------------------------------------------
PREDEFINED_TAGS = """
{
    '#definition': 'a legal statement defining the meaning of concepts [mean, include]',
    '#exclusion': 'a phrase highlighting what is excluded from the definition of a term [exclude, not include]',
    '#exemption': 'a legal statement that exempts someone/something from a rule [exempt, does not apply to, does not require]',
    '#obligation': 'a statement imposing mandatory action to be performed by an agent [shall, must]',
    '#permission': 'a statement indicating the possibility to perform an action without an obligation or a prohibition [may, is permitted to, can, be deemed]',
    '#prohibition': 'a statement forbidding an action to happen or take place [may not, shall not, must not]',
    '#penalty': 'a statement indicating the punishment for not following a rule',
    '#information': 'a legal statement about something that is known or proved to be true',
    '#continuation': 'denoting nested legal statements; assigned whenever a phrase contains a colon and is followed by a bullet list',
    '#condition': 'a phrase in a statement highlighting a constraint under which a rule applies [if, when, after]',
    '#follows': 'relation that connects a statement to references or other statements that precede (act as pre-conditions to) the statement [pursuant to, in accordance with, under]',
    '#refines': 'relation that connects a statement that provides more information about a reference or base statement to the reference or base statement',
    '#followed_by': 'relation that connects a statement to references or other statements that follow the statement',
    '#refined_by': 'relation that connects a base statement to a cross reference or another statement that provides more information about the base statement [as defined in, as described in]',
    '#exception': 'relation that connects a statement to references or other expressions that are exceptions to the statement [unless, except]',
    '#exception_to': 'relation that connects a statement that acts as a exception to a reference or base statement with the reference or base statement [notwithstanding]',
    '#reference': 'when the text contains pointers, numbers, or names to other sections, paragraphs, or laws'
}
"""

TAG_PROMPT = """Read the text and assign tags based on the definitions provided. Do not create your own tags. Only output the tags in the form of a python list. Do not include the assigned parts of the text in your response.

Tag Definitions:
%s

Text: %s
Tags: """

PROMPT_STEP1 = """Read the text and convert it to Python code. Use the class structure detailed below to write code. Do not create your own names. Examples have been provided.

Class Structure:
%s

Examples:
%s

Text: %s
Code: """

PROMPT_STEP2 = """Read the code snippet and decompose Statements into its subclasses. Examples have been provided. Use the class structure detailed below to write code. Do not create your own names.

Class Structure:
%s

Examples:
%s

Original Code: %s
Edited Code: """

PROMPT_STEP3 = """Read the code snippet and identify references and statement relationships (to one another and to external references). Examples have been provided. Use the class structure detailed below to write code. Do not create your own names.

Class Structure:
%s

Examples:
%s

Original Code: %s
Edited Code: """


# ---------------------------------------------------------------------------
# Together client + retrying chat / embedding wrappers
# ---------------------------------------------------------------------------
def make_client() -> Together:
    load_dotenv()
    # Repository root sits one level above this file (code/ -> repo root).
    fallback_env = Path(__file__).resolve().parent.parent / ".env"
    if not os.environ.get("TOGETHER_API_KEY") and fallback_env.exists():
        load_dotenv(fallback_env)
    if not os.environ.get("TOGETHER_API_KEY"):
        sys.exit("TOGETHER_API_KEY not set.")
    return Together()


def _retry_call(fn, *, what: str, max_retries: int = 5):
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return fn()
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            print(f"  retry {attempt + 1}/{max_retries} after {delay:.1f}s "
                  f"({type(e).__name__} during {what})", file=sys.stderr)
            time.sleep(delay); delay *= 2
        except APIError as e:
            status = getattr(e, "http_status", None) or getattr(e, "status_code", None)
            if status and 500 <= int(status) < 600 and attempt < max_retries - 1:
                print(f"  retry {attempt + 1}/{max_retries} after {delay:.1f}s "
                      f"(APIError {status} during {what})", file=sys.stderr)
                time.sleep(delay); delay *= 2
                continue
            raise


def chat(client: Together, model: str, prompt: str, *,
         max_tokens: int = 8192, temperature: float = 0.5,
         enable_thinking: bool = False) -> str:
    # `chat_template_kwargs.enable_thinking` is Qwen3-specific; sending it to
    # other models (Kimi, Llama, DeepSeek) is ignored at best and rejected at
    # worst. Only inject it when the model is from the Qwen family.
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if "qwen" in model.lower():
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    def _call():
        return client.chat.completions.create(**kwargs)
    resp = _retry_call(_call, what=f"chat({model})")
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        content = (getattr(msg, "reasoning", None) or "").strip()
    if not content:
        finish = resp.choices[0].finish_reason
        print(f"  WARNING: empty response (finish={finish}, usage={resp.usage})",
              file=sys.stderr)
    return content


def embed_batch(client: Together, model: str, texts: list,
                batch_size: int = 64, char_cap: int = 1800) -> np.ndarray:
    """Truncate to char_cap before sending. e5/bge family caps at 512 tokens
    (~4 chars/token); 1800 chars is a conservative safe budget."""
    truncated = [t[:char_cap] if isinstance(t, str) else "" for t in texts]
    n_trunc = sum(1 for t, u in zip(texts, truncated)
                  if isinstance(t, str) and len(t) > char_cap)
    if n_trunc:
        print(f"  truncated {n_trunc}/{len(texts)} inputs to {char_cap} chars "
              "(embedding-model context limit)")
    out = []
    for i in range(0, len(truncated), batch_size):
        chunk = truncated[i:i + batch_size]

        def _call(c=chunk):
            return client.embeddings.create(model=model, input=c)
        resp = _retry_call(_call, what=f"embed({model})")
        out.extend([d.embedding for d in resp.data])
        print(f"  embedded {min(i + batch_size, len(truncated))}/{len(truncated)}")
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tag extraction (resilient version of the notebook's parser)
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"#[a-zA-Z_]+")


def extract_tags(answer: str) -> list:
    if not answer:
        return []
    s = answer.strip()
    # Strip ```python ... ``` or ``` ... ``` fences.
    if s.startswith("```"):
        s = re.sub(r"^```(?:python)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Try literal_eval on a clean list-looking substring.
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if m:
        try:
            val = ast.literal_eval(m.group(0))
            if isinstance(val, (list, tuple)):
                return sorted({str(t).strip() for t in val if str(t).strip()})
        except Exception:
            pass
    # Fall back to regex over hash-prefixed tokens.
    return sorted(set(TAG_RE.findall(s)))


# ---------------------------------------------------------------------------
# Demo selection: tag-overlap then cosine similarity, top-10 -> sample 3
# ---------------------------------------------------------------------------
def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return 0.0 if n == 0 else float(np.dot(v1, v2) / n)


def select_demos(test_sample: dict, dev_set: list, n: int = 3) -> list:
    text_tags = set(test_sample.get("tags", []))
    matches = []
    for demo in dev_set:
        overlap = text_tags.intersection(set(demo.get("tags", [])))
        if len(overlap) > 0:
            matches.append([overlap, demo])
    random.shuffle(matches)
    matches.sort(
        key=lambda x: (
            len(x[0]),
            cosine(test_sample["embedding"], x[1]["embedding"]),
        ),
        reverse=True,
    )
    sorted_demos = [m[1] for m in matches]
    if len(sorted_demos) > 10:
        sorted_demos = random.sample(sorted_demos[:10], n)
    return sorted_demos[:n]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def parse_dev_tags(s):
    """Dev-set tags column: stringified python list, '[\"#a\", \"#b\"]'."""
    if isinstance(s, list):
        return s
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(t).strip() for t in v]
    except Exception:
        pass
    # Fall back to the notebook's stripper for legacy rows.
    return [d.strip()[1:-1] for d in s[1:-1].split(",") if d.strip()]


def load_dev(path: Path, seed: int) -> list:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    required = {"text", "code step 1", "code step 2", "code step 3", "tags"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"dev set missing columns: {missing}")
    df["tags"] = df["tags"].apply(parse_dev_tags)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df.to_dict("records")


def load_test(path: Path) -> list:
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace") \
        .to_dict("records")


def tag_test(client: Together, model: str, test_set: list,
             temperature: float, enable_thinking: bool) -> None:
    print(f"tagging {len(test_set)} test segments with {model}")
    for i, t in enumerate(test_set, 1):
        ans = chat(client, model, TAG_PROMPT % (PREDEFINED_TAGS, t["text"]),
                   max_tokens=512, temperature=temperature,
                   enable_thinking=enable_thinking)
        t["tags"] = extract_tags(ans)
        print(f"  [{i}/{len(test_set)}] tags={t['tags']}")


def run_step(client: Together, model: str, prompt_template: str,
             code_string: str, test_set: list, dev_set: list,
             input_key: str, output_key: str, demo_train_key: str,
             temperature: float, max_tokens: int,
             enable_thinking: bool) -> None:
    """input_key='text' for step 1; otherwise the previous step's output column."""
    n = len(test_set)
    for idx, t in enumerate(test_set, 1):
        demos = select_demos(t, dev_set, n=3)
        if not demos:
            demos = random.sample(dev_set, 3)
        if input_key == "text":
            demo_block = "\n\n".join(
                f"Text: {d['text']}\nCode: ```python\n{d[demo_train_key]}\n```"
                for d in demos
            )
            p = prompt_template % (
                "```python\n" + code_string + "\n```",
                demo_block,
                t["text"],
            )
        else:
            demo_block = "\n\n".join(
                f"Original Code: {d[input_key]}\n Edited Code: ```python\n{d[demo_train_key]}\n```"
                for d in demos
            )
            p = prompt_template % (
                "```python\n" + code_string + "\n```",
                demo_block,
                t[input_key],
            )
        t[output_key] = chat(client, model, p, max_tokens=max_tokens,
                             temperature=temperature,
                             enable_thinking=enable_thinking)
        print(f"  [{idx}/{n}] {output_key} ok ({len(t[output_key])} chars)")


def write_step(test_set: list, output_path: Path, code_key: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", code_key, "tags"])
        for t in test_set:
            w.writerow([t["text"], t.get(code_key, ""), t.get("tags", [])])
    print(f"  wrote {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True)
    p.add_argument("--output-dir", default="qwen_results")
    p.add_argument("--law-tag", required=True)
    p.add_argument("--dev-set", default="../development-set.csv")
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    p.add_argument("--tagging-model", default=None,
                   help="Model used for LLM tagging of the test set. "
                        "Defaults to --model.")
    p.add_argument("--embedding-model",
                   default="intfloat/multilingual-e5-large-instruct",
                   help="Together-hosted embedding model. Notebook used "
                        "OpenAI's text-embedding-3-large; switching to a "
                        "Together model is a methodology delta worth noting.")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--thinking", action="store_true",
                   help="Qwen-only: enable thinking mode "
                        "(via chat_template_kwargs). Ignored for non-Qwen models.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", type=int, default=0, metavar="N",
                   help="Smoke test with the first N test segments "
                        "(REQUIRED before any full run).")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tagging_model = args.tagging_model or args.model
    client = make_client()

    dev_set = load_dev(Path(args.dev_set), args.seed)
    test_set = load_test(Path(args.test))
    if args.dry_run > 0:
        test_set = test_set[:args.dry_run]
        print(f"DRY RUN: {len(test_set)} test segments")
    print(f"loaded {len(dev_set)} dev demos, {len(test_set)} test segments")
    print(f"chat model: {args.model}  |  tagging model: {tagging_model}  |  "
          f"embeddings: {args.embedding_model}")

    # 1. Tag test set.
    tag_test(client, tagging_model, test_set, args.temperature, args.thinking)

    # 2. Embed dev + test.
    print("embedding dev set")
    dev_embs = embed_batch(client, args.embedding_model,
                           [d["text"] for d in dev_set])
    for d, e in zip(dev_set, dev_embs):
        d["embedding"] = e
    print("embedding test set")
    test_embs = embed_batch(client, args.embedding_model,
                            [t["text"] for t in test_set])
    for t, e in zip(test_set, test_embs):
        t["embedding"] = e

    # 3. Per-pass three-step decomposition.
    out_dir = Path(args.output_dir)
    model_slug = args.model.split("/")[-1].lower().replace(".", "p")
    suffix = "-dryrun" if args.dry_run > 0 else ""

    for j in range(1, args.passes + 1):
        print(f"\n=== Pass {j}/{args.passes} ===")

        print(" step 1")
        run_step(client, args.model, PROMPT_STEP1, CODE_STRING_STEP1,
                 test_set, dev_set,
                 input_key="text", output_key="code step 1",
                 demo_train_key="code step 1",
                 temperature=args.temperature, max_tokens=args.max_tokens,
                 enable_thinking=args.thinking)
        write_step(test_set,
                   out_dir / f"{args.law_tag}-{model_slug}-step1-{j}{suffix}.csv",
                   "code step 1")

        print(" step 2")
        run_step(client, args.model, PROMPT_STEP2, CODE_STRING_STEP2,
                 test_set, dev_set,
                 input_key="code step 1", output_key="code step 2",
                 demo_train_key="code step 2",
                 temperature=args.temperature, max_tokens=args.max_tokens,
                 enable_thinking=args.thinking)
        write_step(test_set,
                   out_dir / f"{args.law_tag}-{model_slug}-step2-{j}{suffix}.csv",
                   "code step 2")

        print(" step 3")
        run_step(client, args.model, PROMPT_STEP3, CODE_STRING_STEP3,
                 test_set, dev_set,
                 input_key="code step 2", output_key="code step 3",
                 demo_train_key="code step 3",
                 temperature=args.temperature, max_tokens=args.max_tokens,
                 enable_thinking=args.thinking)
        write_step(test_set,
                   out_dir / f"{args.law_tag}-{model_slug}-step3-{j}{suffix}.csv",
                   "code step 3")


if __name__ == "__main__":
    main()
