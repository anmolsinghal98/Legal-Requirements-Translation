"""
Naive-RAG cross-jurisdictional baseline (GPT-5.1, top-15) for the six
queries reported in §8.4 of the REJ paper.

Pipeline:
  1. Embed all 332 corpus paragraph texts with text-embedding-3-large
     (cached to disk in `embeddings_cache.pkl`).
  2. Embed each query.
  3. Retrieve the top-k paragraphs by cosine similarity.
  4. Prompt GPT-5.1 with the retrieved chunks and the query; capture output.

Outputs:
  - `naive_rag.json`  (per-query retrieved set + LLM answer)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from harvest import load_all_laws

HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / 'embeddings_cache.pkl'
OUT_PATH = HERE / 'naive_rag.json'
# Repo root: code/cross_jurisdictional_demo -> code -> repo root
ENV_PATH = str(HERE.parent.parent / '.env')

EMBED_MODEL = 'text-embedding-3-large'
CHAT_MODEL = 'gpt-5.1'
TOP_K = 15

QUESTIONS = {
    'Q1': "Which U.S. states have an explicit obligation to notify the state Attorney General when a data breach of personal information occurs, and what triggering conditions apply?",
    'Q2': "Across the U.S. state data breach notification laws, how does each state define 'personal information', and what categories of data does each state exclude from the definition?",
    'Q3': "Which U.S. states permit a data collector or business to delay notification of a security breach because a law enforcement agency determined that the notification would impede a criminal investigation, and what conditions does each state impose on invoking the delay?",
    'Q4': "Which U.S. state data breach notification laws cross-reference federal statutes such as HIPAA (Health Insurance Portability and Accountability Act), GLBA (Gramm-Leach-Bliley Act), FCRA (Fair Credit Reporting Act), COPPA, or FERPA, and in what context are those federal statutes cited?",
    'Q5': "Which U.S. state data breach notification laws impose a civil penalty with an explicit dollar amount for failure to comply with breach notification obligations, and what is the penalty amount or formula in each state?",
    'Q6': "Which U.S. state data breach notification obligations have explicit exceptions or carve-outs, and what are those exceptions?",
}


def _flatten_paragraphs(laws):
    """Return list of (state, para_idx, text) for all paragraphs across all laws."""
    out = []
    for state, view in laws.items():
        for idx, text in enumerate(view.paragraphs):
            if text.strip():
                out.append((state, idx, text))
    return out


def build_or_load_embeddings(paragraphs):
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('count') == len(paragraphs):
            print(f"Loaded {len(paragraphs)} cached embeddings.")
            return cache['embeddings']
    load_dotenv(ENV_PATH)
    client = OpenAI()
    print(f"Embedding {len(paragraphs)} paragraphs with {EMBED_MODEL} ...")
    texts = [t for _, _, t in paragraphs]
    embs = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embs.extend([d.embedding for d in resp.data])
        print(f"  {i + len(batch)}/{len(texts)}")
    arr = np.array(embs, dtype=np.float32)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'count': len(paragraphs), 'embeddings': arr}, f)
    return arr


def embed_query(question):
    load_dotenv(ENV_PATH)
    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[question])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def cosine_topk(query_vec, corpus, k):
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    cn = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)
    sims = cn @ qn
    idxs = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idxs]


def run_naive_rag(question, paragraphs, embeddings, k=TOP_K, chat_model=CHAT_MODEL):
    qv = embed_query(question)
    hits = cosine_topk(qv, embeddings, k)
    retrieved_lines = []
    retrieved_meta = []
    for i, sim in hits:
        state, para_idx, text = paragraphs[i]
        retrieved_meta.append({'state': state, 'para_idx': para_idx, 'sim': round(sim, 3)})
        retrieved_lines.append(f"[{state} para{para_idx}] {text.strip()}")

    load_dotenv(ENV_PATH)
    client = OpenAI()
    prompt = (
        "You are answering a question about U.S. state data breach notification laws "
        "using only the legal paragraphs below as evidence. List every state that "
        "permits the action described in the question, and for each listed state, "
        "summarize the conditions imposed. If a state is not represented in the "
        "evidence, do not list it.\n\n"
        f"QUESTION: {question}\n\n"
        "EVIDENCE:\n" + "\n\n".join(retrieved_lines) + "\n\n"
        "ANSWER (formatted as a bulleted list: '- <STATE>: <conditions>'):"
    )
    resp = client.chat.completions.create(
        model=chat_model,
        temperature=0.0,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return {
        'question': question,
        'retrieved': retrieved_meta,
        'answer': resp.choices[0].message.content,
        'k': k,
        'chat_model': chat_model,
    }


def run_one(qid, question, paragraphs, embeddings):
    print(f"\n=== {qid}  (model={CHAT_MODEL}, k={TOP_K}) ===")
    res = run_naive_rag(question, paragraphs, embeddings)
    print(f"Retrieved states: {sorted({r['state'] for r in res['retrieved']})}")
    print(f"Answer (first 400 chars):\n{res['answer'][:400]}")
    return res


def main():
    laws = load_all_laws()
    paragraphs = _flatten_paragraphs(laws)
    print(f"Total paragraphs: {len(paragraphs)}")
    embeddings = build_or_load_embeddings(paragraphs)

    # Smoke test on Q3 before committing to the full batch.
    print("[SMOKE] Running Q3 first to validate API contract / parsing.")
    smoke = run_one("Q3", QUESTIONS["Q3"], paragraphs, embeddings)
    if not smoke.get("answer", "").strip():
        sys.exit("SMOKE FAILED: empty answer from " + CHAT_MODEL)
    print("\n[SMOKE OK] Continuing with remaining queries.\n")

    results = {"Q3": smoke}
    for qid, question in QUESTIONS.items():
        if qid == "Q3":
            continue
        results[qid] = run_one(qid, question, paragraphs, embeddings)

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} query outputs to {OUT_PATH}")


if __name__ == '__main__':
    main()
