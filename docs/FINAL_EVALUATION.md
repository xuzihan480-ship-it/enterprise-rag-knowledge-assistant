# Evaluation summary and failure analysis

This is an **internal, synthetic single-PDF evaluation**, not production accuracy or an external reproducibility result.

## Retrieval baselines

20 labelled questions, 48 chunks from a 10-page fictional English handbook. FAISS MRR@8 **0.683**, BM25 **0.938**, Hybrid-RRF **0.867**, FAISS+Cross-Encoder **0.917**, Hybrid+Cross-Encoder **0.967**. The best hybrid method achieved Hit@3 **100%** versus **95%** for FAISS in this small sample. See stored `eval_results/hybrid_summary.json` for exact measurements, constraints, and timings.

## Adaptive reranking

Gate is based on BM25/FAISS top-1 agreement and their relative margin features, **not calibrated confidence probabilities**. Thresholds were chosen using the original ten-question DEV split and kept fixed. On ten HOLDOUT queries: adaptive MRR@8 **1.000** (same as always-rerank), rerank share **40%** (versus 100%), and **estimated** retrieval latency **33.09ms** (versus 75.80ms). The 56.3% decrease is an estimate based on measured components, not direct end-to-end request latency. Consult `eval_results/adaptive_summary.json`.

## Answerability and evidence checks

A separate set of **24 new, manually labelled questions on the same PDF** yielded:

- Answerable accepted: **9 of 12 (75%)**.
- No-answer rejected: **12 of 12 (100%)**.
- Balanced binary decision accuracy: **87.5%**.
- Correct source-page outcome among the 12 positive questions: **9 of 12 (75%)**.

A valid model run completed; these aggregate figures are transcribed from the displayed local final-evaluation summary. The original per-case `verified_challenge.json` and `verified_challenge.csv` were **not present in the previously uploaded source archive**. Therefore this report **does not claim to reproduce raw per-query outcomes**. Add the original machine-generated files from `eval_results/` for transparent review.

### Observed failures and design fixes

1. **false accepts by a similarity-only filter:** legal registration number, current share price, and support telephone number were erroneously accepted because domain overlap did not establish the requested concrete fact.
2. **LLM-generated citation page errors:** one 90-day access-grant answer cited page 3 although its exact source text was on page 2. A provenance-resolver now derives a page from exact matches within the **retrieved passages only**. It does not search other unseen pages to 'repair' evidence.
3. **strict verifier false abstentions:** reranking explanation exists on PDF page 7, but in a final UI check the verifier answered 'no direct answer'. The system is conservative, not perfect.
4. **cost trade-off:** the second local verifier call adds noticeable latency beyond the retrieval phase (a single measured example was ~2.24s for verification); not enough samples exist to claim general latency/SLA values.

The initial six-case diagnostic set informed changes and must not be counted as independent validation. The 24-case set was not used for selecting algorithm thresholds, but was authored after previous error analysis, and is *not an external untouched test set*. Exact quote presence checks provenance; it does not guarantee that the stated answer is logically entailed. Accepted responses require human review of the retrieved quote.

## Next validation milestone

Replicate on multiple independently sourced documents, with separately annotated answerability/grounding labels, per-query latency distributions, reranker ablations, and direct ground-truth comparison of model answers. No such results are claimed here.
