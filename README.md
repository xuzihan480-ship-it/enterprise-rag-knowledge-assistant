# Enterprise Knowledge Assistant

**Local document question-answering with RAG, FAISS, cross-encoder reranking, and vLLM-Metal**

A reproducible **personal engineering demonstration**, built and tested on an Apple M4 MacBook Air (16 GB unified memory). It answers questions about a fictional enterprise handbook using retrieved evidence and returns PDF page references.

> **Scope:** This is a single-PDF, local prototype and a small synthetic retrieval benchmark. It is **not** a production deployment, and the reported evaluation metrics are **not** production accuracy figures.

## Features

- Extract selectable PDF text with `pypdf` and split each page into overlapping chunks (700 characters, 120-character overlap).
- Embed passages and questions with `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieve the **top 8** candidate passages using a normalized-vector FAISS inner-product index (`IndexFlatIP`).
- Rerank candidates with the cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2`; send the **top 3** to the language model.
- Generate answers using the **Qwen2.5-3B-Instruct-4bit** model served **locally** through vLLM-Metal's OpenAI-compatible API.
- Expose `/health` and `/ask` through **FastAPI**, with a local **Streamlit** chat interface displaying source pages and excerpts.
- Compare retrieval ranking before/after reranking using 20 hand-labelled, evidence-anchored questions.

### Architecture

```text
             Fictional 10-page PDF
                     |
               pypdf extraction
                     |
         700-character overlapping chunks
                     |
         SentenceTransformer embeddings
                     |
              FAISS IndexFlatIP
                     |
User question -> embedding -> top-8 candidate retrieval
                     |
             Cross-Encoder rerank
                     |
            Top-3 evidence passages
                     |
      FastAPI -> vLLM-Metal local API (port 8000)
                     |
             Qwen2.5-3B response
                     |
   Streamlit UI (port 8501): answer + page references
```

## Evaluated retrieval results

**Benchmark:** One fictional 10-page PDF; 48 indexed chunks; 20 questions, each with a manually specified source-page and exact evidence anchor.

| Measure | FAISS | FAISS + reranking |
| --- | ---: | ---: |
| Hit@3 | 95.0% | 95.0% |
| MRR@8 | 0.683 | 0.917 |
| Average retrieval time per question | 5.15 ms | 78.29 ms |

- Absolute MRR@8 increase: **0.233**; relative increase: **~34.1%** on this benchmark.
- Correct evidence moved to a better rank for **9/20** questions; rank was unchanged for the other 11.
- The initially retrieved top-8 candidates contained the gold evidence for **19/20** questions. Reranking cannot recover an answer missing from that candidate set.
- Reranking added approximately **73.14 ms/query** on the evaluation machine.
- These are **retrieval-only** metrics; they do not measure generated-answer correctness, statistical significance, or production performance.

See [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md), [`eval_results/summary.json`](eval_results/summary.json), and [`eval_results/retrieval_comparison.csv`](eval_results/retrieval_comparison.csv) for the method, results, and limitations.

## Project structure

```text
enterprise_knowledge_assistant/
├── app.py                         # FastAPI retrieval/rerank/generation endpoint
├── web_ui.py                      # Streamlit chat interface
├── evaluate.py                    # 20-query retrieval evaluation
├── pdf_retrieval.py               # Standalone PDF retrieval example
├── retrieval.py                   # Standalone four-document prototype
├── data/
│   ├── enterprise_knowledge_base_demo.pdf
│   └── company_knowledge.txt
├── eval_results/
│   ├── retrieval_comparison.csv
│   └── summary.json
├── docs/
│   ├── EVALUATION_REPORT.md
│   ├── RUN_COMMANDS.md
│   └── RESUME_PROJECT_CN.md
├── requirements.txt
└── .gitignore
```

The fictional PDF is provided as **demonstration content**, not actual company data. The main application currently indexes a fixed PDF on startup; the UI does not offer document uploads.

## Run locally

### 1. Python environment

Use **Python 3.12 or 3.13** in a clean virtual environment on macOS. The machine used for this project used Python 3.13 for the retrieval/API/UI environment.

```bash
cd enterprise_knowledge_assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some Apple Silicon environments have native-library threading issues. For API and evaluation runs, limit the relevant CPU threads as shown below.

### 2. Start vLLM-Metal (separate environment, terminal A)

Install vLLM-Metal by following the official project's current installation instructions: https://github.com/vllm-project/vllm-metal

In the setup tested for this project, the dedicated environment was located at `~/.venv-vllm-metal`. Once installed, run:

```bash
~/.venv-vllm-metal/bin/vllm serve mlx-community/Qwen2.5-3B-Instruct-4bit \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.4 --max-model-len 2048
```

Do not close this terminal while testing the chat application. Validate the service from another terminal:

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

**Important:** This project validated **vLLM-Metal on Apple Silicon**, **not a CUDA/NVIDIA deployment**. A separate Modal/L4 experiment was attempted but was blocked because the service required adding a payment method; it is not included as a completed deployment.

### 3. Start FastAPI (terminal B)

```bash
cd enterprise_knowledge_assistant
source .venv/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Check:
- Health: http://127.0.0.1:8001/health
- Interactive API docs: http://127.0.0.1:8001/docs

Example request:

```bash
curl -sS http://127.0.0.1:8001/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the five data quality dimensions?"}'
```

### 4. Start Streamlit (terminal C)

```bash
cd enterprise_knowledge_assistant
source .venv/bin/activate
python -m streamlit run web_ui.py --server.port 8501
```

Open http://localhost:8501.

### 5. Reproduce the retrieval benchmark

The following commands do **not** need vLLM, FastAPI, Streamlit, or a cloud GPU.

```bash
cd enterprise_knowledge_assistant
source .venv/bin/activate
python evaluate.py --validate-only

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false python evaluate.py
```

The script regenerates `eval_results/summary.json` and `eval_results/retrieval_comparison.csv`. Timings can vary between runs and computers.

## Current limitations and next steps

1. **Synthetic, single-document benchmark:** Evaluated on a fictional handbook with 20 questions; it is not a real enterprise workload.
2. **Chunk quality:** Fixed-length character splits can cut sentences in half and duplicate overlapping content; page numbers are tracked but chunks lack stable semantic boundaries.
3. **Retrieval misses:** One test question (Q06) did not retrieve the labelled evidence in the first eight candidates. Reranking cannot correct an initial retrieval miss.
4. **Generation quality:** Small language models can give incomplete answers even when relevant context is retrieved. Manual checks found both successes and omissions; automated answer correctness was not measured.
5. **Limited deployment scope:** The server binds to localhost, uses a static PDF and hard-coded model name, and has no authentication, rate limits, or production monitoring.
6. **Document safety:** Retrieved text is treated as reference material, but this demo has no fully evaluated prompt-injection defence or access-control system.

Potential improvements: section-aware or token-aware chunking, multiple-document indexing, evaluation on an independent held-out set, automated answer-grounding tests, and additional failure-case analysis.

## Reproducibility and attribution

The included `eval_results` files are actual saved results from the local evaluation. The result summary identifies model versions by repository ID but does not freeze downloaded model revisions, so very small differences may occur on rerun.

No private company materials, paid APIs, published papers, production SLAs, or cloud GPU deployments are claimed by this project.
