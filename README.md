# Adaptive Enterprise Knowledge Assistant

**Evidence-aware document Q&A using hybrid retrieval, adaptive reranking and local Qwen2.5 inference.**

A working **local prototype** developed and tested on an Apple M4 MacBook Air (16 GB). It indexes a **fictional 10-page English enterprise handbook** (48 overlapping text chunks), retrieves evidence and generates short answers with source-page references—or explicitly abstains when evidence is insufficient. This repository is a research-and-engineering demonstration, **not an operational enterprise system**.

## Architecture

```text
Fictional PDF -> pypdf text extraction -> overlapping chunks
                                         |                           BM25 keyword search  FAISS semantic search (MiniLM)
                                         \ /
                                     RRF fusion (top-8)
                                            |
                              Adaptive rerank gate (DEV-tuned)
                                  /                                                  skip                Cross-Encoder
                                  \                    /
                                     Top-3 evidence
                                            |
                         Evidence sufficiency heuristic (DEV-tuned)
                                            |
                         Qwen2.5-3B verifier via local vLLM-Metal
                           (specific-answer check + verbatim quote)
                                            |
                           Exact quote matched against retrieved text
                              -> derive/correct page or abstain
                                            |
                          FastAPI on :8001 -> Streamlit on :8503
```

**Important:** Verbatim quote matching establishes **provenance**, not logical entailment. The 3B verifier can misjudge semantic support, and strict checks can incorrectly reject answerable questions. A failed model-service call is treated as an error, **not** a successful refusal.

## Highlights

- **Hybrid retrieval:** in-project BM25 implementation + FAISS dense search + reciprocal rank fusion (RRF).
- **Adaptive reranking:** compare top-1 agreement and relative score margins to selectively invoke a MiniLM cross-encoder; thresholds selected using the original **10-question DEV** subset and held fixed on the 10-question HOLDOUT.
- **Evidence-aware answers:** a lightweight evidence filter, followed by a local model that checks for direct support and returns an exact supporting quote. Code derives page numbers from the retrieved passage rather than trusting model-generated page numbers.
- **Integrated UI and API:** FastAPI `GET /health`, `POST /ask`; Streamlit chat with retrieval route, citations and refusal reasons.
- **Reproducible scripts:** algorithm comparisons, DEV/HOLDOUT analysis and answerability challenge tests (with explicit limitations).

## Measured results (local / synthetic)

### Retrieval comparison — 20 labelled questions, one 10-page PDF, 48 chunks

| Method | Hit@3 | MRR@8 | Mean retrieval time |
|:--|--:|--:|--:|
| FAISS | 95% | 0.683 | 5.03 ms |
| BM25 | 95% | 0.938 | 0.07 ms |
| FAISS + Cross-Encoder | 95% | 0.917 | 79.09 ms |
| Hybrid RRF | 100% | 0.867 | 5.10 ms |
| **Hybrid + Cross-Encoder** | **100%** | **0.967** | 76.58 ms |

See [`eval_results/hybrid_summary.json`](eval_results/hybrid_summary.json). This experiment measures **retrieval ranking, not generated-answer accuracy**. BM25 is strong on this lexical-overlap-heavy synthetic dataset.

### Adaptive reranking — 10 DEV + 10 HOLDOUT questions

| HOLDOUT method | Hit@3 | MRR@8 | Rerank usage | Estimated retrieval cost |
|:--|--:|--:|--:|--:|
| Hybrid RRF | 100% | 0.900 | 0% | 5.08 ms/query |
| Hybrid + Cross-Encoder | 100% | 1.000 | 100% | 75.80 ms/query |
| **Adaptive** | **100%** | **1.000** | **40%** | **33.09 ms/query** |

The adaptive gate reduced reranker usage by **60%** in this **10-question** holdout. Its **56.3% estimated retrieval-cost reduction** comes from measured component costs—not a direct API response-time benchmark. See [`eval_results/adaptive_summary.json`](eval_results/adaptive_summary.json).

### Evidence-aware QA — 24 new internal challenge questions

| Measure | Observed |
|:--|--:|
| Answerable questions accepted | **9/12 (75%)** |
| Unanswerable questions rejected | **12/12 (100%)** |
| Balanced answerability decision accuracy | **87.5%** |
| Correct source-page result among 12 answerable cases | **9/12 (75%)** |

These aggregate figures were recorded from the local final evaluation run. They reflect **decision outcomes and source-page checks, not independent answer factuality review**. The 24 cases use the *same fictional PDF* and were authored after investigating earlier errors; this is **not an independent external benchmark**. The strict verifier sometimes falsely abstains, including on a question about reranking. See [`docs/FINAL_EVALUATION.md`](docs/FINAL_EVALUATION.md). Raw final-run `verified_challenge.json` and `.csv` were saved on the Mac but were **not present in the available project ZIP**; they should be added separately if publishing per-question evidence.

## Run locally on Apple Silicon

The retrieval/API/UI environment and vLLM-Metal inference environment are **separate**. vLLM-Metal here serves a quantized **Qwen2.5-3B-Instruct-4bit** model, **not** CUDA vLLM or a paid GPU cloud deployment.

1. Create the retrieval environment (Python 3.12 or 3.13):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Install vLLM-Metal separately, following its [official project](https://github.com/vllm-project/vllm-metal). For the existing setup, terminal A:

```bash
~/.venv-vllm-metal/bin/vllm serve mlx-community/Qwen2.5-3B-Instruct-4bit \
  --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.4 --max-model-len 2048
```

3. Terminal B (API):

```bash
cd enterprise-rag-knowledge-assistant
source .venv/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false python -m uvicorn app_verified:app --host 127.0.0.1 --port 8001
```

4. Terminal C (UI):

```bash
cd enterprise-rag-knowledge-assistant
source .venv/bin/activate
python -m streamlit run web_ui_verified.py --server.port 8503
```

Open `http://localhost:8503` or `http://127.0.0.1:8001/docs`.

Example questions:

- Answerable: `What are the five data quality dimensions?` (handbook page 3)
- Answerable: `How long does a standard access grant last?` (page 2)
- Unanswerable: `What is the official customer support telephone number?` (should abstain)

A correct page is **not** evidence of overall factuality by itself. These services bind to localhost and are not internet-hosted.

## Reproduce the tests

Offline validation and unit checks (no running LLM required):

```bash
python evaluate_verified.py --validate-only
python -m unittest discover -s tests -p 'test_verified_core.py'
python answer_support.py
```

Run with vLLM on :8000 (prefer stopping the UI/API first on 16 GB memory):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false python evaluate_hybrid.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false python evaluate_adaptive.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false python evaluate_verified.py
```

`evaluate_verified.py --smoke` runs 4 diagnostic cases, **not** the final 24-case benchmark. Scores are small-sample descriptive results; model downloads, hardware and version drift can affect timings and outputs.

## Repository organization

```text
app_verified.py, verified_rag.py         # Final FastAPI / retrieval-and-verification pipeline
web_ui_verified.py                      # Final Streamlit chat
hybrid_retrieval.py, adaptive_rerank.py  # Retrieval/fusion and adaptive gate
answer_support.py, evidence_guard.py    # Evidence checks and quote/page resolution
evaluate_hybrid.py, evaluate_adaptive.py, evaluate_verified.py
                                       # Separate evaluation scripts
data/verified_challenge_cases.json      # 24 hand-checked internal test labels
eval_results/hybrid_summary.json, adaptive_summary.json
                                       # Stored retrieval metrics
docs/FINAL_EVALUATION.md               # Evaluation interpretation and limitations
tests/test_verified_core.py             # Pure offline regression checks
app.py, web_ui.py, evaluate.py          # Original baseline kept for comparison
```

## Limitations

- Fixed *fictional* 10-page PDF and manually authored questions; no multi-document or cross-domain evidence of generalization.
- A small 10-question adaptive HOLDOUT is not proof of robust deployment-level cost improvement.
- Quote matching is not semantic entailment validation; the LLM verifier can return wrong judgments or hallucinated answers.
- Conservative evidence checking yields false abstentions; **75%** of answerable challenge queries were accepted.
- Local-only deployment: no authentication, access control, API hardening, ingestion UI, production monitoring, or cloud GPU benchmark.
- Verifier latency is additional to retrieval latency. Measured single-query timings must not be treated as full-system latency distributions.
- RAG prompt injection, adversarial robustness and data confidentiality are not comprehensively evaluated.

All demo content is fictional. Project implementation, measured trade-offs and failure analysis are documented without claims of original theoretical novelty, publications, or production deployment.
