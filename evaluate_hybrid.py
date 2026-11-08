"""Benchmark FAISS, BM25, FAISS+Rerank, Hybrid RRF, Hybrid+Rerank.

Reads the existing PDF, chunking and 20 evidence annotations from evaluate.py.
Run from the same directory as evaluate.py:
    python evaluate_hybrid.py --validate-only
    python evaluate_hybrid.py

Outputs NEW CSV/JSON files without overwriting the previous experiment.
All scores are for a small synthetic 10-page handbook, NOT production claims.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from evaluate import TESTS, load_chunks, validate_labels
from hybrid_retrieval import BM25Index, rrf_fuse, self_test

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "eval_results"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_RERANK = "cross-encoder/ms-marco-MiniLM-L6-v2"
CANDIDATE_K = 8
METRIC_K = 3
METHODS = ("FAISS", "BM25", "FAISS+Rerank", "Hybrid-RRF", "Hybrid+Rerank")


def gold_rank(ids: list[int], gold: set[int]) -> int:
    """1-based rank of the first gold chunk, or 0 if none retrieved."""
    return next((rank for rank, chunk_id in enumerate(ids, 1) if chunk_id in gold), 0)


def run(chunks: list[dict], gold_by_id: dict[str, set[int]]) -> None:
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print("Loading embedding model (CPU)...", flush=True)
    embedder = SentenceTransformer(MODEL_EMBEDDING, device="cpu")
    print("Loading reranking model (CPU)...", flush=True)
    reranker = CrossEncoder(MODEL_RERANK, device="cpu")

    texts = [chunk["text"] for chunk in chunks]
    bm25 = BM25Index(texts)
    vectors = embedder.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Warm up model kernels before recording query latency.
    embedder.encode(["Warmup question"], normalize_embeddings=True)
    reranker.predict([["Warmup question", texts[0]]])
    print(f"Index ready: {len(chunks)} chunks, {len(TESTS)} questions.\n", flush=True)

    def cross_encoder_rank(question: str, ids: list[int]) -> tuple[list[int], float]:
        start = perf_counter()
        scores = reranker.predict([[question, texts[i]] for i in ids])
        order = sorted(range(len(ids)), key=lambda j: (-float(scores[j]), j))
        return [ids[j] for j in order], perf_counter() - start

    rows = []
    for test in TESTS:
        question = test["question"]
        gold = gold_by_id[test["id"]]

        t0 = perf_counter()
        q_vec = embedder.encode([question], normalize_embeddings=True).astype("float32")
        _, positions = index.search(q_vec, min(CANDIDATE_K, len(chunks)))
        faiss_ids = [int(i) for i in positions[0]]
        faiss_time = perf_counter() - t0

        t0 = perf_counter()
        bm25_ids = bm25.search(question, min(CANDIDATE_K, len(chunks)))
        bm25_time = perf_counter() - t0

        t0 = perf_counter()
        hybrid_ids = rrf_fuse(faiss_ids, bm25_ids, top_k=CANDIDATE_K)
        fuse_time = perf_counter() - t0

        faiss_reranked, faiss_rerank_time = cross_encoder_rank(question, faiss_ids)
        hybrid_reranked, hybrid_rerank_time = cross_encoder_rank(question, hybrid_ids)

        rankings = {
            "FAISS": (faiss_ids, faiss_time),
            "BM25": (bm25_ids, bm25_time),
            "FAISS+Rerank": (faiss_reranked, faiss_time + faiss_rerank_time),
            "Hybrid-RRF": (hybrid_ids, faiss_time + bm25_time + fuse_time),
            "Hybrid+Rerank": (
                hybrid_reranked,
                faiss_time + bm25_time + fuse_time + hybrid_rerank_time,
            ),
        }
        for method in METHODS:
            ids, duration = rankings[method]
            rank = gold_rank(ids, gold)
            rows.append({
                "question_id": test["id"],
                "question": question,
                "gold_page": test["page"],
                "method": method,
                "gold_rank_at_8": rank,
                "hit_at_3": int(0 < rank <= METRIC_K),
                "mrr_at_8": 1.0 / rank if rank else 0.0,
                "hit_at_8": int(rank > 0),
                "latency_ms": round(duration * 1000, 3),
                "top3_pages": "|".join(str(chunks[i]["page"]) for i in ids[:3]),
                "top8_ids": "|".join(map(str, ids)),
            })
        print(
            f"{test['id']}: FAISS {gold_rank(faiss_ids, gold) or '-'} | "
            f"BM25 {gold_rank(bm25_ids, gold) or '-'} | "
            f"RRF {gold_rank(hybrid_ids, gold) or '-'} | "
            f"RRF+Rerank {gold_rank(hybrid_reranked, gold) or '-'}",
            flush=True,
        )

    summary = {}
    for method in METHODS:
        part = [row for row in rows if row["method"] == method]
        summary[method] = {
            "hit_at_3": mean(row["hit_at_3"] for row in part),
            "mrr_at_8": mean(row["mrr_at_8"] for row in part),
            "candidate_hit_at_8": mean(row["hit_at_8"] for row in part),
            "mean_latency_ms": mean(row["latency_ms"] for row in part),
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "hybrid_comparison.csv"
    json_path = OUTPUT / "hybrid_summary.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({
        "benchmark": "20 hand-labelled questions; 10-page fictional PDF; 48 original chunks",
        "candidates_per_retriever": CANDIDATE_K,
        "rrf_k": 60,
        "metric_k": METRIC_K,
        "methods": summary,
        "limitations": [
            "This is a small synthetic benchmark, not an external generalisation test.",
            "Questions and gold anchors may share words with the source; lexical search can benefit.",
            "All methods use the same PDF chunks and source-evidence labels.",
            "RRF fuses each retriever's top 8 then keeps the top 8 for reranking.",
            "Generation quality, refusal behaviour and factuality are NOT evaluated here.",
            "Mean timings exclude initial model loading/index construction and hardware vary.",
        ],
    }, indent=2), encoding="utf-8")

    print("\n================ HYBRID RETRIEVAL BENCHMARK ================")
    print(f"{'Method':<20} {'Hit@3':>8} {'MRR@8':>8} {'Hit@8':>8} {'ms/query':>11}")
    for method in METHODS:
        s = summary[method]
        print(f"{method:<20} {s['hit_at_3']:>7.1%} {s['mrr_at_8']:>8.3f} "
              f"{s['candidate_hit_at_8']:>7.1%} {s['mean_latency_ms']:>11.2f}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print("NOTE: Small fictional benchmark; report measured results, including regressions.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    self_test()
    chunks = load_chunks()
    gold = validate_labels(chunks)
    print(f"Validated {len(TESTS)} gold questions against {len(chunks)} chunks.")
    if not args.validate_only:
        run(chunks, gold)


if __name__ == "__main__":
    main()
