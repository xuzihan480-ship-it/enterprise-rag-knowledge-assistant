# Retrieval Evaluation and Failure Analysis

**Project:** Enterprise Knowledge Assistant  
**Date:** 8 November 2026  
**Evidence:** `eval_results/summary.json`, `eval_results/retrieval_comparison.csv`, `evaluate.py`  
**Scope:** Local, small synthetic benchmark; **retrieval evaluation only**

## 1. Objective

Evaluate whether adding a cross-encoder reranker improves the ordering of evidence passages compared with embedding-based FAISS retrieval alone, and quantify the latency cost. The purpose is to test the actual application retrieval pipeline, **not** to claim an improvement to generation accuracy or performance on enterprise data.

## 2. Experimental setup

| Component | Configuration |
| --- | --- |
| Knowledge base | Fictional 10-page English enterprise handbook |
| Indexed chunks | 48 |
| Chunking | Per-page, 700 characters per chunk, 120-character overlap |
| Evaluation questions | 20 manually written questions with gold page and exact supporting phrase |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, CPU, L2-normalized |
| Vector search | `faiss.IndexFlatIP` (inner product of normalized embeddings) |
| Initial candidate count | 8 passages per question |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2`, CPU |
| Retrieval context budget | Top 3 ranked passages sent to generation in `app.py` |
| Compared methods | Same eight FAISS candidates, before vs after reranking |
| Timing | Per-question embedding+FAISS search and additional rerank prediction, using `time.perf_counter` |

**Gold-label method:** Each question has a source PDF page and a verbatim supporting phrase. The script first validates that the exact phrase occurs in at least one chunk from the stated page. A candidate is marked relevant if it is one of those evidence-containing chunks. This avoids using the retriever's own output as ground truth.

**Methodological limitation:** These gold labels identify **one required evidence fact**, not every passage that might be semantically relevant or all evidence required for a long answer. The report therefore calls the top-3 metric **Hit@3**; under this single-fact target formulation it is equivalent to Recall@3.

## 3. Metric definitions

For each question, let *r* be the rank of the first gold-evidence chunk; an unavailable rank contributes zero.

- **Candidate Hit@8:** Fraction of questions whose gold evidence is in the initial eight FAISS candidates.
- **Hit@3:** Fraction whose gold evidence is among the first three ranked passages.
- **MRR@8:** Mean reciprocal rank of the first gold-evidence chunk, truncated to the first eight candidates.
- **Latency:** Mean CPU embedding+FAISS search time, plus the separately measured reranking time, in the environment used for this saved run.

## 4. Results

| Metric | FAISS baseline | FAISS + Cross-Encoder |
| --- | ---: | ---: |
| Hit@3 | **95.0%** | **95.0%** |
| MRR@8 | **0.6833** | **0.9167** |
| Mean retrieval time | **5.15 ms/query** | **78.29 ms/query** |

Additional findings:

- Candidate Hit@8: **95.0%** (**19/20** queries).
- MRR@8 improved by **0.2333 absolute**, or approximately **34.1% relative** to the baseline, on this benchmark.
- Gold evidence moved up in rank on **9/20** queries; remained at the same rank on **11/20**; moved down on **0/20**.
- Reranking added **73.14 ms per query** on average in this run.
- At top 3, both methods achieved 19/20 hits. The improvement was primarily **better ordering**, not the ability to discover new evidence.

**Interpretation:** A reranker can improve the ordering of already-retrieved passages; it cannot add passages omitted by the initial retrieval stage. The latency increase is substantial relative to this tiny index, and its practical value should be tested under realistic loads.

## 5. Specific cases

| Query | Initial gold rank | Reranked gold rank | Observation |
| --- | ---: | ---: | --- |
| Q02 — Restricted-data approvals | 2 | 1 | Better ordering |
| Q05 — Failed quality check | 2 | 1 | Better ordering |
| Q16 — Ranking vs latency trade-off | 2 | 1 | Better ordering |
| Q20 — Cold-start vs warm latency | 3 | 1 | Better ordering |
| Q01 — Standard access duration | 3 | 3 | No change |
| **Q06 — Daily-data freshness requirement** | **Not in top 8** | **Not in top 8** | **Initial retrieval miss** |

The Q06 miss is an actionable failure case. The evidence exists in the PDF and was validated as a gold chunk, but FAISS did not retrieve it among the top eight candidates. Reranking cannot fix this. Possible follow-up experiments include different embedding models, query rephrasing, section-aware chunking, and different candidate counts; these have **not** yet been tested.

## 6. Manual answer-generation checks

The live demonstration used a local **Qwen2.5-3B-Instruct-4bit** model served via **vLLM-Metal**, with a FastAPI endpoint and Streamlit interface. The screenshots supplied during development support these **qualitative observations**, not a formal answer-accuracy score:

- The question asking for the five data-quality dimensions produced all five names and explanations with a PDF page reference when tested with the 3B model and explicit list instruction.
- The question about a failed quality check produced a document-consistent explanation mentioning issue logging and quarantining the new dataset release.
- The question about “Accepted with Conditions” correctly identified the central criterion but did not include all supporting requirements (such as remediation owner and review date).
- The explanation of high raw agreement correctly identified the class-imbalance problem.
- The explanation of reranking matched the cited passage on cross-encoder evaluation and latency.

**Important control issue:** During development, both the model size (0.5B to 3B) **and** the wording of the five-dimensions question changed. Therefore the better answer **cannot be attributed solely to the model upgrade**. No controlled model-comparison experiment was performed.

## 7. Main limitations

1. **Small, authored synthetic dataset:** One 10-page document and 20 questions with known facts. Results are not generalisable to real corporate documents or unseen questions.
2. **Single-source retrieval:** No multi-document retrieval, corpus updates, hybrid search, or access-controlled knowledge sources.
3. **Chunk boundaries:** Character-based splitting may cut in the middle of words or sentences and produce overlapping passages.
4. **Gold-label simplification:** Exact evidence anchors are useful for reproducibility, but only a single target fact is evaluated per query.
5. **Timing:** Reported latencies are measured locally on a small CPU retrieval workload; they exclude PDF preprocessing, initial model loading, vLLM answer generation, UI rendering, concurrency, and real network overhead.
6. **Generation quality:** There is no completed automated faithfulness, hallucination, citation-correctness, or unanswerable-question benchmark.
7. **Deployment:** The tested deployment is a **local Apple Silicon vLLM-Metal service**, not a Linux/CUDA cloud GPU or publicly accessible production service.
8. **No statistical uncertainty:** The 20-question sample is too small for reliable general claims; no confidence intervals or independent holdout study were reported.

## 8. Reproduction

From the project root:

```bash
source .venv/bin/activate
python evaluate.py --validate-only
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false python evaluate.py
```

Outputs are saved to `eval_results/retrieval_comparison.csv` and `eval_results/summary.json`. The saved result snapshot is included in this repository. Slight latency differences are expected across runs.

## 9. Appropriate summary for interviews

> On a small 20-question synthetic document-retrieval benchmark, cross-encoder reranking raised MRR@8 from 0.683 to 0.917 while Hit@3 remained at 95%. It added about 73 ms of retrieval latency per question. The result shows improved evidence ordering on this dataset, not a verified increase in production retrieval or answer accuracy.
