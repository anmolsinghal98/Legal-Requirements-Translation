# Legal Requirements Translation from Law

![License](https://img.shields.io/github/license/anmolsinghal98/Legal-Requirements-Translation)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Models](https://img.shields.io/badge/LLMs-GPT--4o%20%7C%20GPT--5.1%20%7C%20GPT--5.3--Codex%20%7C%20Qwen3.5-lightgrey)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15794182.svg)](https://doi.org/10.5281/zenodo.15794182)

## 📝 Summary of the Artifact

Ensuring software compliance with legal regulations is especially challenging for small organizations. Extracting legal requirements from lengthy, complex texts demands significant legal expertise.

Prior automated approaches often overlook the **interdependencies** between legal metadata attributes. Yet legal meaning frequently arises from these relationships—for example, how obligations, conditions, and exceptions interact across clauses and sections.

This artifact provides a **framework for translating legal requirements into executable Python code**. It packages the original GPT-4o single-shot pipeline from the IEEE RE 2025 paper of Singhal and Breaux *together with* the extensions submitted to the Requirements Engineering Journal (REJ):

- A **decomposition-based pipeline** that breaks the text-to-code task into three sub-tasks (structure → statement classification → references/relationships).
- Decomposition runners for **GPT-5.3-Codex** (OpenAI Responses API) and **Qwen3.5-397B-A17B** (Together AI), in addition to the GPT-4o / GPT-5.1 notebooks.
- A new **COPPA test set** (60 paragraphs from 16 CFR Part 312) extending the RE 2025 corpus of six U.S. state data breach laws.
- A consolidated **`code/evaluation/` harness** (compilation, structural, semantic, pass@k, tag prediction) that scores any directory of generated CSVs.

### ✨ Key Features

- **Two translation pipelines**
  - *Standard (single-shot)*: GPT-4o code generation with embedding-based few-shot example selection (RE 2025).
  - *Decomposition (3-step)*: progressive in-context learning that reveals successively more of the class structure at each step (REJ extension).
- **Domain-specific class structure** with bidirectional relationship enforcement (`refines` ↔ `is_refined_by`, `has_exception` ↔ `is_exception_to`, `follows` ↔ `is_followed_by`).
- **Reusable test harness** — 1 compilation + 5 structural + 16 semantic attribute tests applied to each generated paragraph.

---

## 📂 Repository Layout

```
Legal-Requirements-Translation/
├── code/
│   ├── class_structure.py              # Core metamodel
│   ├── serialize.py                    # Object → dict serialization
│   ├── test_statements.py              # 22-test unit-test harness (shared by both eval forms)
│   │
│   ├── Code-with-Demo-with-Class.ipynb # Standard (single-shot) pipeline (RE 2025)
│   ├── Code-with-Decomposition.ipynb   # Decomposition (3-step) pipeline (REJ extension)
│   │
│   ├── run_codex_decomposition.py      # Decomposition runner — GPT-5.3-Codex (Responses API)
│   ├── run_qwen_decomposition.py       # Decomposition runner — Qwen3.5 via Together AI
│   ├── run_all_laws_codex.sh           # Drives Codex decomposition across all test laws
│   ├── run_all_laws_decomp.sh          # Drives Qwen decomposition across all test laws
│   │
│   ├── evaluation/                     # Evaluation harness (REJ) + the published RE 2025 notebooks
│   │   ├── eval_lib.py                 # Compilation, structural, semantic, pass@k library
│   │   ├── eval_tags.py                # Tag-prediction evaluation (Appendix E of REJ)
│   │   ├── run_eval.py                 # CLI entrypoint
│   │   ├── run_all_evals.sh
│   │   ├── Code-Gen-Compliation-Testing.ipynb
│   │   ├── Code-Gen-Structural-Testing.ipynb
│   │   ├── Code-Gen-Semantic-Testing.ipynb
│   │   └── Compute-pass-at-k.ipynb
│   │
│   ├── cross_jurisdictional_demo/      # Cross-jurisdictional retrieval demo (REJ §8.4)
│   │   ├── harvest.py                  # Harvest Rule/Definition/Exemption/Information/Reference instances from each law's ground-truth code
│   │   ├── queries.py                  # Six structured cross-jurisdictional queries over the harvested representation
│   │   ├── naive_rag.py                # Naive-RAG baseline (GPT-5.1 + top-15) for the same six queries
│   │   └── corpus/                     # 13-state ground-truth corpus (`<state>-training.csv`, 332 paragraphs)
│   │
│   └── intermediate-results/           # Sample outputs for the MS test law (smoke-test fixture)
│
├── test files/                         # Held-out evaluation set
│   ├── OR.csv  MS.csv  VA.csv  VT.csv  UT.csv  WI.csv     # RE 2025 test laws
│   └── COPPA.csv                                           # REJ extension (16 CFR Part 312)
│
├── development-set.csv                 # Development corpus (paragraph + step-1/2/3 ground-truth code + tags)
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## 🧱 Core Class Structure

The metamodel in `code/class_structure.py` encodes legal metadata at three levels of granularity:

| Class | Role |
|---|---|
| `Section` | A bullet point in the legal text; nests `subSections`; contains `expressions` and `statements`. |
| `Expression` | A text snippet within a `Section`; the leaf-level textual unit. |
| `Reference(Expression)` | An `Expression` that points to another `Section` or `Statement`, with a `relationship` label. |
| `Statement` | Spans one or more `Section`s. Six bidirectional relationships: `refines`/`is_refined_by`, `has_exception`/`is_exception_to`, `follows`/`is_followed_by`. |
| `Information` | Factual statement (`description`). |
| `Definition` | Defines a term (`defined_term`, `meaning`, `exclusions`). |
| `Rule` | `rule_type` (OBLIGATION=0, PERMISSION=1, PROHIBITION=2, PENALTY=3), `entity`, `description`, `conditions`. |
| `Exemption` | Exemption statement (`description`). |

The decomposition pipeline progressively widens the class context shown to the LLM at each step (see `Code-with-Decomposition.ipynb`); step 1 inlines a reduced version exposing only `Section`, `Expression`, and `Statement`.

---

## 🔁 Pipelines

### 1. Standard (single-shot) — `code/Code-with-Demo-with-Class.ipynb`

For each input paragraph:

1. Embed the paragraph with `text-embedding-3-large`.
2. Retrieve the top-3 nearest neighbours from `development-set.csv` after a tag-overlap pre-filter.
3. Issue a single GPT-4o prompt that includes the full class structure and the three exemplars, asking the model to emit Python code instantiating the class structure.

### 2. Decomposition (3-step) — `code/Code-with-Decomposition.ipynb`

The same retrieval step is followed by three successive prompts that progressively expose more of the class structure:

| Step | Class context shown to the LLM | Output |
|---|---|---|
| 1 | `Section`, `Expression`, `Statement` (base classes only) | Hierarchical structure with generic `Statement` placeholders |
| 2 | + `Rule`, `Definition`, `Exemption`, `Information` | Each `Statement` replaced with the correct subclass and its attributes |
| 3 | + `Reference` and inter-statement relationships | `Reference` instances and `refines` / `has_exception` / `follows` links added |

Each step's exemplars are tailored: step-1 exemplars show only base classes, step-2 exemplars show classified statements, etc. The intermediate ground-truth labels live in the `code step 1`, `code step 2`, and `code step 3` columns of `development-set.csv`.

### 3. Decomposition runners (CLI)

For batch reproduction without Jupyter, two runners cover the open- and closed-source decomposition experiments reported in the REJ paper. Run them from `code/`:

```bash
cd code

# GPT-5.3-Codex decomposition over the COPPA test set, 3 passes
python run_codex_decomposition.py \
    --test "../test files/COPPA.csv" \
    --law-tag COPPA --passes 3

# Qwen3.5-397B-A17B decomposition (Together AI)
python run_qwen_decomposition.py \
    --test "../test files/COPPA.csv" \
    --law-tag COPPA --passes 3

# All test laws, 3 passes each:
bash run_all_laws_codex.sh        # GPT-5.3-Codex   -> code/codex_results/
bash run_all_laws_decomp.sh       # Qwen3.5         -> code/qwen_results/
```

Each pass writes three CSVs per law — one per decomposition step (`<law>-<model>-step{1,2,3}-<pass>.csv`) — under the runner's `--output-dir`.

> **Smoke-test first.** Always validate new model integrations with `--dry-run 3` (3 paragraphs) before launching a full run.

The GPT-4o and GPT-5.1 decomposition results in the paper were generated from `Code-with-Decomposition.ipynb` rather than a runner; see the notebook for prompt and post-processing details.

---

## ✅ Evaluation Harness (`code/evaluation/`)

`code/evaluation/` consolidates everything required to score generated code. There are two equivalent forms:

| Form | Files | When to use |
|---|---|---|
| **CLI library (REJ)** — *maintained, default* | `eval_lib.py`, `run_eval.py`, `run_all_evals.sh`, `eval_tags.py` | Batch evaluation over many (model, law, pass) cells; produces `evaluation_output/<model>/summary.csv` and an aggregated `all_models_summary.csv`. |
| **Notebook walkthroughs (RE 2025)** | `Code-Gen-Compliation-Testing.ipynb`, `Code-Gen-Structural-Testing.ipynb`, `Code-Gen-Semantic-Testing.ipynb`, `Compute-pass-at-k.ipynb` | Cell-by-cell explanation of the test design; matches the published RE 2025 walkthrough. |

Both forms call into the same source-of-truth in `code/test_statements.py`, so the scores are identical.

```bash
cd code/evaluation

# Score one model:
python run_eval.py \
    --model-tag codex_decomp \
    --gen-dir ../codex_results \
    --pattern '{law}-gpt-5p3-codex-step3-{pass}.csv' \
    --passes 3

# Or score every shipped decomposition variant and aggregate:
bash run_all_evals.sh
```

`eval_tags.py` reproduces the tag-prediction analysis (Appendix E of REJ); supply the manually-coded ground-truth tag CSV via `--gt`.

---

## 🌐 Cross-Jurisdictional Demo (`code/cross_jurisdictional_demo/`)

Reproduces §8.4 of the REJ paper: shows how the executable code representation supports structured cross-jurisdictional queries, and compares that against a naive-RAG baseline over the same corpus.

The demo operates over the full 13-state ground-truth corpus (332 paragraphs) shipped under `corpus/`. `harvest.py` `exec`s each ground-truth Python snippet in an isolated namespace and collects the resulting `Rule`, `Definition`, `Exemption`, `Information`, and `Reference` instances, tagging each with `(state, paragraph_idx)`. `queries.py` defines the six structured cross-jurisdictional queries (Q1–Q6) that operate directly on those harvested instances.

```bash
cd code/cross_jurisdictional_demo

# Sanity check: harvest the full corpus and print per-state element counts (no API calls).
python harvest.py

# Run the six structured queries against the harvested representation:
python queries.py

# Run the same six queries through the naive-RAG baseline (GPT-5.1 + top-15).
# The script first runs Q3 as a smoke test, then continues with Q1, Q2, Q4–Q6.
python naive_rag.py                      # -> naive_rag.json
```

`naive_rag.py` reads `OPENAI_API_KEY` from the repository-root `.env`. Paragraph embeddings are cached on disk (`embeddings_cache.pkl`) so subsequent runs only re-embed if the corpus changes.

---

## 🧪 Test Set (`test files/`)

CSV files in `test files/` contain legal paragraphs and their hand-authored Python translations:

| File | Law | Source |
|---|---|---|
| `OR.csv` | Oregon | RE 2025 |
| `MS.csv` | Mississippi | RE 2025 |
| `VA.csv` | Virginia | RE 2025 |
| `VT.csv` | Vermont | RE 2025 |
| `UT.csv` | Utah | RE 2025 |
| `WI.csv` | Wisconsin | RE 2025 |
| `COPPA.csv` | Children's Online Privacy Protection Act, 16 CFR Part 312 | **REJ extension** |

Although tailored for U.S. state and federal privacy laws, the class structure is domain-agnostic and adapts to other legal domains.

`code/intermediate-results/` ships with sample outputs for the MS test law so the evaluation harness can be exercised end-to-end without making any API calls.

---

## ⚙️ System Requirements

**Hardware**:
- CPU: ≥ 2 cores
- RAM: 8 GB minimum (16 GB recommended)
- Disk: ≤ 100 MB

**Software**:
- OS: Linux/macOS/Windows
- Python: 3.10 or higher (CPython, the reference interpreter for the Compilation Test)
- Tools: Git, Jupyter

**API access**:
- `OPENAI_API_KEY` — required for GPT-4o, GPT-5.1, and GPT-5.3-Codex.
- `TOGETHER_API_KEY` — required for Qwen3.5-397B-A17B.

---

## 🔧 Installation

### 1. Clone

```bash
git clone https://github.com/anmolsinghal98/Legal-Requirements-Translation.git
cd Legal-Requirements-Translation
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the repository root:

```
OPENAI_API_KEY=your_openai_key
TOGETHER_API_KEY=your_together_key   # only if running Qwen
```

OpenAI key instructions: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key

### 4. Launch Jupyter (for the notebook-based pipeline)

```bash
jupyter notebook
```

---

## 🚀 Usage

1. **Translate** with `code/Code-with-Demo-with-Class.ipynb` (single-shot, GPT-4o) or `code/Code-with-Decomposition.ipynb` (decomposition, GPT-4o/GPT-5.1).
2. **Generate at scale** with `run_codex_decomposition.py` or `run_qwen_decomposition.py`. Always pass `--dry-run 3` first, then run a full pass.
3. **Score** with `code/evaluation/run_eval.py`.

### ⏱ Avoiding multi-hour runs

A full re-execution across every (model, law, pass) cell takes longer than 60 minutes due to API latency. Sample outputs for the MS test law are committed under `code/intermediate-results/` so the evaluation harness (notebook or CLI) can be exercised end-to-end without API calls.

---

## 🔁 Steps to Reproduce

1. **Development-set results.** Run the chosen translation notebook over `development-set.csv`, score with `code/evaluation/run_eval.py`, and average across passes.
2. **Test-set results.** Run the chosen pipeline over each file in `test files/` and score with `code/evaluation/run_eval.py`. Aggregate per-model results into `all_models_summary.csv`.
3. **Determinism.** GPT-4o, GPT-5.1, and GPT-5.3-Codex are closed-source and may drift; numbers reported in the paper can shift slightly between OpenAI model snapshots, but the qualitative findings should remain valid. Sampling temperature is fixed at 0.5 across all models.
4. **Post-processing edge cases.** Generated outputs occasionally contain stray whitespace or characters that break the compilation test. Comments in the runners and notebooks flag the lines most likely to need adjustment for new model versions.

---

## 🔄 Using Other Language Models

The prompts are model-agnostic. The runners are organised around a clear seam:

- OpenAI-hosted chat/Responses-API models — extend `run_codex_decomposition.py`.
- Together-AI-hosted open-source models — extend `run_qwen_decomposition.py`.
- Self-hosted HuggingFace models — install the optional `transformers` and `torch` dependencies and adapt the same module.

Performance across models can vary substantially with parameter count, instruction tuning, and code-generation training data. Numbers reported in the paper should not be assumed to transfer.

---

## 👥 Author Information

1. Anmol Singhal — Carnegie Mellon University (anmolsinghal@cmu.edu)
2. Travis Breaux — Carnegie Mellon University (tdbreaux@andrew.cmu.edu)

## 🌍 Artifact Location

The artifact is archived at: https://doi.org/10.5281/zenodo.15794182

## 📚 How to Cite

### RE 2025 (original)

A. Singhal, T.D. Breaux (2025). *Legal Requirements Translation from Law.* 33rd IEEE International Requirements Engineering Conference.

### REJ extension

A. Singhal, T.D. Breaux. *Decomposing Legal Requirements Translation: A Multi-Model Code-Based Pipeline for Regulatory Compliance.* Requirements Engineering Journal (under review).

You can also use the metadata in `CITATION.cff`.

## 📄 License

See the [LICENSE](LICENSE) file for details (Apache-2.0).
