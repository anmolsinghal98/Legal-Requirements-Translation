"""
Decomposition pipeline (3-step) on OpenAI's Responses API, default model
gpt-5.3-codex. Mirrors Code-with-Decomposition.ipynb exactly: LLM tagging,
embedding-based retrieval (text-embedding-3-large), three-step prompting
with progressive class structures.

Reuses prompt strings, demo-selection, and IO helpers from
run_qwen_decomposition.py so the only differences are the API surface.

Each pass writes three CSVs under --output-dir:
  <law>-<model_slug>-step1-<pass>.csv
  <law>-<model_slug>-step2-<pass>.csv
  <law>-<model_slug>-step3-<pass>.csv

Usage:
  python run_codex_decomposition.py \
      --test "../test files/COPPA.csv" \
      --law-tag COPPA --dry-run 3
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    APIError,
)

from run_qwen_decomposition import (
    CODE_STRING_STEP1,
    CODE_STRING_STEP2,
    CODE_STRING_STEP3,
    PREDEFINED_TAGS,
    TAG_PROMPT,
    PROMPT_STEP1,
    PROMPT_STEP2,
    PROMPT_STEP3,
    extract_tags,
    select_demos,
    load_dev,
    load_test,
    write_step,
)


# ---------------------------------------------------------------------------
# OpenAI client + retrying Responses / Embeddings wrappers
# ---------------------------------------------------------------------------
def make_client() -> OpenAI:
    load_dotenv()
    # Repository root sits one level above this file (code/ -> repo root).
    fallback = Path(__file__).resolve().parent.parent / ".env"
    if not os.environ.get("OPENAI_API_KEY") and fallback.exists():
        load_dotenv(fallback)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set.")
    # Per-request timeout so a hung connection surfaces as APITimeoutError
    # (which the retry loop handles) instead of blocking forever.
    return OpenAI(timeout=180.0, max_retries=0)


def _retry_call(fn, *, what: str, max_retries: int = 5):
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return fn()
        except (RateLimitError, APITimeoutError,
                APIConnectionError, InternalServerError) as e:
            if attempt == max_retries - 1:
                raise
            print(f"  retry {attempt + 1}/{max_retries} after {delay:.1f}s "
                  f"({type(e).__name__} during {what})", file=sys.stderr)
            time.sleep(delay); delay *= 2
        except APIError as e:
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            if status and 500 <= int(status) < 600 and attempt < max_retries - 1:
                print(f"  retry {attempt + 1}/{max_retries} after {delay:.1f}s "
                      f"(APIError {status} during {what})", file=sys.stderr)
                time.sleep(delay); delay *= 2
                continue
            raise


def respond(client: OpenAI, model: str, prompt: str, *,
            max_output_tokens: int = 8192,
            reasoning_effort: str = "medium",
            temperature: float | None = None) -> str:
    """One Responses API call. Reasoning models burn output tokens on
    reasoning, so the cap defaults higher than the chat-completions version."""
    kwargs = dict(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": reasoning_effort},
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    def _call():
        return client.responses.create(**kwargs)
    resp = _retry_call(_call, what=f"responses({model})")

    text = (getattr(resp, "output_text", None) or "").strip()
    if not text:
        # Fall back to walking the output array for any text content.
        chunks = []
        for item in (getattr(resp, "output", None) or []):
            for c in (getattr(item, "content", None) or []):
                t = getattr(c, "text", None)
                if t:
                    chunks.append(t)
        text = "\n".join(chunks).strip()
    if not text:
        status = getattr(resp, "status", None)
        usage = getattr(resp, "usage", None)
        print(f"  WARNING: empty response (status={status}, usage={usage})",
              file=sys.stderr)
    return text


def embed_batch(client: OpenAI, model: str, texts: list,
                batch_size: int = 64, char_cap: int = 30000) -> np.ndarray:
    """text-embedding-3-large supports 8191 tokens (~32K chars); we cap
    defensively but truncation should be rare on these segments."""
    truncated = [t[:char_cap] if isinstance(t, str) else "" for t in texts]
    n_trunc = sum(1 for t, u in zip(texts, truncated)
                  if isinstance(t, str) and len(t) > char_cap)
    if n_trunc:
        print(f"  truncated {n_trunc}/{len(texts)} inputs to {char_cap} chars")
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
# Pipeline glue (codex-specific run_step + tag_test)
# ---------------------------------------------------------------------------
def tag_test(client: OpenAI, model: str, test_set: list,
             reasoning_effort: str, temperature: float | None) -> None:
    print(f"tagging {len(test_set)} test segments with {model}")
    for i, t in enumerate(test_set, 1):
        ans = respond(client, model,
                      TAG_PROMPT % (PREDEFINED_TAGS, t["text"]),
                      max_output_tokens=4096,
                      reasoning_effort=reasoning_effort,
                      temperature=temperature)
        t["tags"] = extract_tags(ans)
        print(f"  [{i}/{len(test_set)}] tags={t['tags']}")


def run_step(client: OpenAI, model: str, prompt_template: str,
             code_string: str, test_set: list, dev_set: list,
             input_key: str, output_key: str, demo_train_key: str,
             max_output_tokens: int, reasoning_effort: str,
             temperature: float | None) -> None:
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
        t[output_key] = respond(client, model, p,
                                max_output_tokens=max_output_tokens,
                                reasoning_effort=reasoning_effort,
                                temperature=temperature)
        print(f"  [{idx}/{n}] {output_key} ok ({len(t[output_key])} chars)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True)
    p.add_argument("--output-dir", default="codex_results")
    p.add_argument("--law-tag", required=True)
    p.add_argument("--dev-set", default="../development-set.csv")
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--start-pass", type=int, default=1,
                   help="Resume from this pass (1-indexed). Tagging and "
                        "embeddings always re-run.")
    p.add_argument("--model", default="gpt-5.3-codex")
    p.add_argument("--tagging-model", default=None,
                   help="Model used for LLM tagging. Defaults to --model.")
    p.add_argument("--embedding-model", default="text-embedding-3-large")
    p.add_argument("--reasoning-effort",
                   choices=["low", "medium", "high", "xhigh"],
                   default="low")
    p.add_argument("--temperature", type=float, default=None,
                   help="Optional. Reasoning models often ignore this; "
                        "leave unset to use the model default.")
    p.add_argument("--max-output-tokens", type=int, default=8192,
                   help="Includes reasoning tokens.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", type=int, default=0, metavar="N")
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
    print(f"chat model: {args.model} (effort={args.reasoning_effort})  |  "
          f"tagging: {tagging_model}  |  embeddings: {args.embedding_model}")

    tag_test(client, tagging_model, test_set,
             args.reasoning_effort, args.temperature)

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

    out_dir = Path(args.output_dir)
    model_slug = args.model.replace("/", "-").replace(".", "p").lower()
    suffix = "-dryrun" if args.dry_run > 0 else ""

    for j in range(args.start_pass, args.passes + 1):
        print(f"\n=== Pass {j}/{args.passes} ===")

        print(" step 1")
        run_step(client, args.model, PROMPT_STEP1, CODE_STRING_STEP1,
                 test_set, dev_set,
                 input_key="text", output_key="code step 1",
                 demo_train_key="code step 1",
                 max_output_tokens=args.max_output_tokens,
                 reasoning_effort=args.reasoning_effort,
                 temperature=args.temperature)
        write_step(test_set,
                   out_dir / f"{args.law_tag}-{model_slug}-step1-{j}{suffix}.csv",
                   "code step 1")

        print(" step 2")
        run_step(client, args.model, PROMPT_STEP2, CODE_STRING_STEP2,
                 test_set, dev_set,
                 input_key="code step 1", output_key="code step 2",
                 demo_train_key="code step 2",
                 max_output_tokens=args.max_output_tokens,
                 reasoning_effort=args.reasoning_effort,
                 temperature=args.temperature)
        write_step(test_set,
                   out_dir / f"{args.law_tag}-{model_slug}-step2-{j}{suffix}.csv",
                   "code step 2")

        print(" step 3")
        run_step(client, args.model, PROMPT_STEP3, CODE_STRING_STEP3,
                 test_set, dev_set,
                 input_key="code step 2", output_key="code step 3",
                 demo_train_key="code step 3",
                 max_output_tokens=args.max_output_tokens,
                 reasoning_effort=args.reasoning_effort,
                 temperature=args.temperature)
        write_step(test_set,
                   out_dir / f"{args.law_tag}-{model_slug}-step3-{j}{suffix}.csv",
                   "code step 3")


if __name__ == "__main__":
    main()
