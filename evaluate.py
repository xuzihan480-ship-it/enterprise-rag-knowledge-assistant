"""Evaluate FAISS vs FAISS + Cross-Encoder reranking on 20 PDF questions.

Designed for the Enterprise Knowledge Assistant demonstrated on a MacBook.
The PDF, chunking strategy, and models match the existing app.py.

Usage (in the project folder):
    python evaluate.py --validate-only
    python evaluate.py

The 20 gold labels below were created from statements in the demonstration PDF.
Each label identifies a specific supporting phrase and the PDF page, rather
than trusting the retriever to decide which result is correct.

No vLLM, internet API, GPU, or cloud account is needed to score retrieval.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "data" / "enterprise_knowledge_base_demo.pdf"
OUTPUT_DIR = ROOT / "eval_results"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CANDIDATE_K = 8  # Match the existing app.py: FAISS returns eight candidates.
METRIC_K = 3  # The app sends the top three reranked chunks to the LLM.

# Manually specified gold evidence, checked against the source PDF.
# page: original PDF page (1-indexed); anchor: verbatim supporting passage.
# The passage must fit inside at least one 700-character chunk.
TESTS = [
    {"id": "Q01", "question": "How long does a standard access grant last?", "page": 2,
     "anchor": "Standard access grants expire after 90 calendar days"},
    {"id": "Q02", "question": "Who must additionally approve access to Restricted data?", "page": 2,
     "anchor": "For Restricted data, security approval is additionally mandatory."},
    {"id": "Q03", "question": "Can Restricted data be copied onto personal devices?", "page": 2,
     "anchor": "Restricted data must not be copied to personal devices."},
    {"id": "Q04", "question": "What are the five data quality dimensions?", "page": 3,
     "anchor": "completeness, uniqueness, schema conformity, freshness, and consistency."},
    {"id": "Q05", "question": "What happens to the new dataset release if a quality check fails?", "page": 3,
     "anchor": "The pipeline marks the new data release as quarantined."},
    {"id": "Q06", "question": "How quickly must new daily data become available?", "page": 3,
     "anchor": "New daily data available within 12 hours"},
    {"id": "Q07", "question": "What is the default sampling rate for routine dataset evaluation?", "page": 4,
     "anchor": "the default sample is 10% of available records"},
    {"id": "Q08", "question": "How much of the sample is reviewed by a second annotator?", "page": 4,
     "anchor": "At least 20% of the sample is independently reviewed by a second annotator."},
    {"id": "Q09", "question": "What must be specified when data is Accepted with Conditions?", "page": 4,
     "anchor": "Acceptance with Conditions must specify a remediation owner and review date."},
    {"id": "Q10", "question": "How many answerable and unanswerable questions are in the pilot benchmark?", "page": 5,
     "anchor": "40 answerable questions and 10 intentionally unanswerable questions."},
    {"id": "Q11", "question": "What evidence-recall decrease triggers a release investigation?", "page": 5,
     "anchor": "Any decrease greater than 0.03 absolute points in Evidence Recall@5 requires investigation"},
    {"id": "Q12", "question": "Does a reranker score of 7.5 mean 75 percent confidence?", "page": 5,
     "anchor": "a score of 7.5 does not mean 75% confidence."},
    {"id": "Q13", "question": "What chunk size and overlap does the handbook recommend?", "page": 6,
     "anchor": "350 tokens per chunk with a 60-token overlap."},
    {"id": "Q14", "question": "How are embedding vectors prepared for FAISS cosine-similarity search?", "page": 6,
     "anchor": "Vectors are L2-normalized before insertion into a FAISS inner-product index"},
    {"id": "Q15", "question": "How does a cross-encoder evaluate question and passage relevance?", "page": 7,
     "anchor": "A cross-encoder evaluates the question and each candidate passage together"},
    {"id": "Q16", "question": "What is the ranking benefit and latency drawback of reranking?", "page": 7,
     "anchor": "This can improve ranking precision but increases latency."},
    {"id": "Q17", "question": "What must the assistant say if the handbook cannot answer a question?", "page": 8,
     "anchor": "the assistant must say that the available handbook does not specify the answer."},
    {"id": "Q18", "question": "Should an instruction embedded in a retrieved PDF override the application rules?", "page": 8,
     "anchor": "Retrieved document text is untrusted input."},
    {"id": "Q19", "question": "What OpenAI-compatible chat-completions URL path does vLLM expose?", "page": 9,
     "anchor": "commonly addressed at /v1/chat/completions."},
    {"id": "Q20", "question": "Why distinguish cold-start and warm-request response latency?", "page": 10,
     "anchor": "Monitoring must distinguish cold-start latency from warm-request latency"},
]


def load_chunks():
    """Use exactly the PDF chunking logic from the existing FastAPI app."""
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Cannot find {PDF_PATH}\nPlace the test PDF in the project's data/ folder.")
    reader = PdfReader(str(PDF_PATH))
    chunks = []
    for page_num, page in enumerate(reader.pages, 1):
        text = " ".join((page.extract_text() or "").split())
        start = 0
        while start < len(text):
            end = min(start + 700, len(text))
            section = text[start:end].strip()
            if len(section) >= 80:
                chunks.append({"id": len(chunks), "page": page_num, "text": section})
            if end == len(text):
                break
            start = end - 120
    if not chunks:
        raise ValueError("No searchable text found in the PDF.")
    return chunks


def validate_labels(chunks):
    """Check that every human-labelled evidence phrase really occurs in a chunk."""
    if len(TESTS) != 20 or len({t["id"] for t in TESTS}) != 20:
        raise ValueError("Exactly twenty unique benchmark questions are required.")
    gold = {}
    for test in TESTS:
        matches = {
            ch["id"] for ch in chunks
            if ch["page"] == test["page"]
            and test["anchor"].casefold() in ch["text"].casefold()
        }
        if not matches:
            raise ValueError(
                f"Gold evidence missing for {test['id']} (page {test['page']}): "
                f"{test['anchor']!r}\nThe PDF or chunking may have changed."
            )
        gold[test["id"]] = matches
    return gold


def first_relevant_rank(ordered_chunk_ids, gold_chunk_ids):
    for rank, chunk_id in enumerate(ordered_chunk_ids, 1):
        if chunk_id in gold_chunk_ids:
            return rank
    return None


def mean(nums):
    return sum(nums) / len(nums) if nums else 0.0


def run_evaluation(chunks, gold):
    # Heavy imports happen only for the full benchmark, not --validate-only.
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"Loading {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    print(f"Loading {RERANK_MODEL}...")
    reranker = CrossEncoder(RERANK_MODEL, device="cpu")

    vectors = embedder.encode(
        [ch["text"] for ch in chunks], normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    print(f"Indexed {len(chunks)} PDF chunks. Starting {len(TESTS)} questions...\n")

    records = []
    for test in TESTS:
        question = test["question"]
        # Baseline: embedding computation + FAISS search.
        t0 = time.perf_counter()
        query_vec = embedder.encode(
            [question], normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32")
        _, positions = index.search(query_vec, min(CANDIDATE_K, len(chunks)))
        baseline_ids = [chunks[int(i)]["id"] for i in positions[0]]
        baseline_seconds = time.perf_counter() - t0

        # Rerank EXACTLY the same initial candidates; no different retrieval set.
        t1 = time.perf_counter()
        pairs = [[question, chunks[i]["text"]] for i in baseline_ids]
        scores = reranker.predict(pairs, show_progress_bar=False)
        sorted_positions = sorted(
            range(len(baseline_ids)),
            key=lambda j: (-float(scores[j]), j),
        )
        reranked_ids = [baseline_ids[j] for j in sorted_positions]
        rerank_extra_seconds = time.perf_counter() - t1

        gold_ids = gold[test["id"]]
        base_rank = first_relevant_rank(baseline_ids, gold_ids)
        rank_rank = first_relevant_rank(reranked_ids, gold_ids)
        base_hit = int(base_rank is not None and base_rank <= METRIC_K)
        rerank_hit = int(rank_rank is not None and rank_rank <= METRIC_K)

        record = {
            "id": test["id"],
            "question": question,
            "gold_page": test["page"],
            "gold_anchor": test["anchor"],
            "gold_chunk_ids": "|".join(map(str, sorted(gold_ids))),
            "candidate_hit_at_8": int(base_rank is not None),
            "faiss_first_gold_rank": base_rank or 0,
            "rerank_first_gold_rank": rank_rank or 0,
            "faiss_hit_at_3": base_hit,
            "rerank_hit_at_3": rerank_hit,
            "faiss_mrr_at_8": 1 / base_rank if base_rank else 0.0,
            "rerank_mrr_at_8": 1 / rank_rank if rank_rank else 0.0,
            "faiss_top3_pages": "|".join(str(chunks[i]["page"]) for i in baseline_ids[:METRIC_K]),
            "rerank_top3_pages": "|".join(str(chunks[i]["page"]) for i in reranked_ids[:METRIC_K]),
            "faiss_seconds": round(baseline_seconds, 4),
            "rerank_extra_seconds": round(rerank_extra_seconds, 4),
            "total_with_rerank_seconds": round(baseline_seconds + rerank_extra_seconds, 4),
        }
        records.append(record)
        status = "IMPROVED" if rerank_hit > base_hit else (
            "WORSE" if rerank_hit < base_hit else "SAME"
        )
        print(
            f"{test['id']}: FAISS gold rank {base_rank or '-':>2} | "
            f"Rerank gold rank {rank_rank or '-':>2} | {status}"
        )
    return records


def save_results(chunks, records):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "retrieval_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "fictional 10-page enterprise handbook; 20 hand-labelled queries",
        "pdf_filename": PDF_PATH.name,
        "questions": len(records),
        "pdf_chunks": len(chunks),
        "candidate_k": CANDIDATE_K,
        "metric_k": METRIC_K,
        "embedding_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "faiss_hit_at_3": mean([r["faiss_hit_at_3"] for r in records]),
        "rerank_hit_at_3": mean([r["rerank_hit_at_3"] for r in records]),
        "faiss_mrr_at_8": mean([r["faiss_mrr_at_8"] for r in records]),
        "rerank_mrr_at_8": mean([r["rerank_mrr_at_8"] for r in records]),
        "candidate_hit_at_8": mean([r["candidate_hit_at_8"] for r in records]),
        "mean_faiss_seconds": mean([r["faiss_seconds"] for r in records]),
        "mean_rerank_extra_seconds": mean([r["rerank_extra_seconds"] for r in records]),
        "mean_total_with_rerank_seconds": mean([r["total_with_rerank_seconds"] for r in records]),
        "notes": [
            "Both methods use identical top-8 FAISS candidates; reranking only changes order.",
            "Hit@3 equals Recall@3 for a query treated as one required evidence fact.",
            "Gold facts were hand-labelled using page + exact supporting evidence phrase.",
            "MRR@8 is calculated within the initial top-8 retrieval set.",
            "The benchmark measures retrieval only, NOT generated-answer correctness.",
            "Data are synthetic and small; reported scores cannot generalise to production.",
        ],
    }
    json_path = OUTPUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n========== RETRIEVAL EVALUATION ==========")
    print(f"Questions: {summary['questions']} | PDF chunks: {summary['pdf_chunks']}")
    print(f"Candidate hit@8 : {summary['candidate_hit_at_8']:.1%}")
    print(f"FAISS Hit@3     : {summary['faiss_hit_at_3']:.1%}")
    print(f"Rerank Hit@3    : {summary['rerank_hit_at_3']:.1%}")
    print(f"FAISS MRR@8     : {summary['faiss_mrr_at_8']:.3f}")
    print(f"Rerank MRR@8    : {summary['rerank_mrr_at_8']:.3f}")
    print(f"Mean FAISS time : {summary['mean_faiss_seconds']:.3f} s/query")
    print(f"Rerank extra    : {summary['mean_rerank_extra_seconds']:.3f} s/query")
    print(f"Mean full time  : {summary['mean_total_with_rerank_seconds']:.3f} s/query")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    print("Scores are from a SMALL SYNTHETIC benchmark, not a production accuracy claim.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FAISS retrieval against reranking.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the PDF and human-labelled evidence only.")
    args = parser.parse_args()
    chunks = load_chunks()
    gold = validate_labels(chunks)
    print(f"Validated {len(TESTS)} gold questions against {len(chunks)} text chunks.")
    if args.validate_only:
        print("Gold labels OK; no AI models loaded. Ready to run python evaluate.py")
        return
    records = run_evaluation(chunks, gold)
    save_results(chunks, records)


if __name__ == "__main__":
    main()
