"""Evaluate adaptive rerank gating against Hybrid-RRF and Hybrid+Rerank.

No change to the existing app.py; experiments remain isolated.

Run in project folder (after copying both stage-2 Python files):
    python evaluate_adaptive.py --validate-only
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false python evaluate_adaptive.py

Fairness:
- A fixed split: odd numbered questions = DEV, even = HOLDOUT.
- All gate thresholds are selected on DEV only; HOLDOUT is not used in tuning.
- Skip/re-rank decision uses only retrieval-time features, never gold labels.
- The quality report has separate DEV and HOLDOUT rows.
- Adaptive latency is an estimate assembled from measured retrieval/rerank components.
  It is not a directly timed deployed FastAPI endpoint.
- Tiny synthetic benchmark. No generalization or production claims.
"""
import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from adaptive_rerank import Gate, gate_features, skip_rerank, self_test as gate_self_test
from evaluate import TESTS, load_chunks, validate_labels
from evaluate_hybrid import gold_rank
from hybrid_retrieval import BM25Index, rrf_fuse, self_test as hybrid_self_test

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "eval_results"
EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
K = 8

# Deliberately small, fixed, human-readable grid. Not a machine-learning model.
BM25_THRESHOLDS = (0.0, 0.05, 0.10, 0.20, 0.40)
COSINE_THRESHOLDS = (0.0, 0.02, 0.04, 0.08, 0.12)
DEV_MRR_TOLERANCE = 0.02


def split_of(test_id: str) -> str:
    return "DEV" if int(test_id[1:]) % 2 else "HOLDOUT"


def evaluate_policy(records: list[dict], gate: Gate, split: str):
    part = [r for r in records if r["split"] == split]
    predictions = []
    for r in part:
        skipped = skip_rerank(r["features"], gate)
        rank = r["rrf_rank"] if skipped else r["rerank_rank"]
        predictions.append({
            "id": r["id"],
            "skipped": skipped,
            "rank": rank,
            "hit_at_3": int(0 < rank <= 3),
            "mrr_at_8": 1 / rank if rank else 0.0,
            # Component-based latency estimate; not direct app request timing.
            "estimated_ms": r["retrieval_ms"] + (0.0 if skipped else r["rerank_ms"]),
        })
    return {
        "n": len(predictions),
        "hit_at_3": mean(p["hit_at_3"] for p in predictions),
        "mrr_at_8": mean(p["mrr_at_8"] for p in predictions),
        "rerank_rate": mean(not p["skipped"] for p in predictions),
        "estimated_latency_ms": mean(p["estimated_ms"] for p in predictions),
    }, predictions


def baseline_summary(records: list[dict], split: str, method: str):
    part = [r for r in records if r["split"] == split]
    key = "rerank_rank" if method == "Hybrid+Rerank" else "rrf_rank"
    lat_key = "retrieval_ms" if method == "Hybrid-RRF" else "total_rerank_ms"
    return {
        "n": len(part),
        "hit_at_3": mean(0 < r[key] <= 3 for r in part),
        "mrr_at_8": mean(1/r[key] if r[key] else 0.0 for r in part),
        "rerank_rate": 0.0 if method == "Hybrid-RRF" else 1.0,
        "estimated_latency_ms": mean(r[lat_key] for r in part),
    }


def tune_gate(records: list[dict]):
    """Select fastest DEV-safe gate; test labels do not influence selection."""
    dev_full = baseline_summary(records, "DEV", "Hybrid+Rerank")
    valid = []
    for lex in BM25_THRESHOLDS:
        for sem in COSINE_THRESHOLDS:
            gate = Gate(lex, sem)
            s, _ = evaluate_policy(records, gate, "DEV")
            if (
                s["hit_at_3"] >= dev_full["hit_at_3"] - 1e-10
                and s["mrr_at_8"] + 1e-10 >= dev_full["mrr_at_8"] - DEV_MRR_TOLERANCE
            ):
                valid.append((s["rerank_rate"], -s["mrr_at_8"], lex, sem, gate))
    if valid:
        return min(valid, key=lambda row: row[:4])[-1], True
    # Safe fallback: gate can never trigger at these thresholds.
    return Gate(10.0, 10.0), False


def run(chunks: list[dict], gold: dict[str, set[int]]):
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    texts = [c["text"] for c in chunks]
    print("Loading local embedding and cross-encoder models...", flush=True)
    embedder = SentenceTransformer(EMBEDDING, device="cpu")
    reranker = CrossEncoder(RERANKER, device="cpu")
    bm25 = BM25Index(texts)
    vectors = embedder.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    embedder.encode(["Warmup question"], normalize_embeddings=True)
    reranker.predict([["Warmup question", texts[0]]])

    records = []
    for test in TESTS:
        question = test["question"]
        t0 = perf_counter()
        qvec = embedder.encode([question], normalize_embeddings=True).astype("float32")
        dense_scores, dense_indices = index.search(qvec, K)
        dense_ids = [int(x) for x in dense_indices[0]]
        # BM25 top scores are computed independently, not compared with cosine scores.
        lex_scores_all = bm25.score(question)
        lex_ids = sorted(range(len(texts)), key=lambda i: (-lex_scores_all[i], i))[:K]
        hybrid_ids = rrf_fuse(dense_ids, lex_ids, top_k=K)
        retrieval_ms = (perf_counter() - t0) * 1000

        features = gate_features(
            dense_ids, [float(x) for x in dense_scores[0]],
            lex_ids, [float(lex_scores_all[i]) for i in lex_ids],
        )
        t1 = perf_counter()
        scores = reranker.predict([[question, texts[i]] for i in hybrid_ids])
        rerank_ms = (perf_counter() - t1) * 1000
        order = sorted(range(len(hybrid_ids)), key=lambda j: (-float(scores[j]), j))
        reranked_ids = [hybrid_ids[j] for j in order]

        rrf_rank = gold_rank(hybrid_ids, gold[test["id"]])
        rerank_rank = gold_rank(reranked_ids, gold[test["id"]])
        records.append({
            "id": test["id"],
            "split": split_of(test["id"]),
            "question": question,
            "gold_page": test["page"],
            "features": features,
            "rrf_rank": rrf_rank,
            "rerank_rank": rerank_rank,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "total_rerank_ms": retrieval_ms + rerank_ms,
        })
        print(f"{test['id']} ({split_of(test['id'])}): "
              f"RRF rank {rrf_rank or '-'} -> Rerank rank {rerank_rank or '-'} | "
              f"top-1 agree: {features['top1_agreement']}", flush=True)

    chosen_gate, found = tune_gate(records)
    print("\nTuned using DEV only:", asdict(chosen_gate))
    if not found:
        print("No tested gate satisfied DEV quality constraints. Using always-rerank fallback.")
    summary = {
        "benchmark": "Small synthetic 10-page PDF / 48 chunks / 20 gold-labeled questions",
        "split": "Odd question IDs = DEV (10), even IDs = HOLDOUT (10)",
        "chosen_gate": asdict(chosen_gate),
        "selected_on": "DEV only; HOLDOUT was not used for threshold selection",
        "threshold_grid": {"bm25": BM25_THRESHOLDS, "cosine": COSINE_THRESHOLDS},
        "dev_mrr_tolerance": DEV_MRR_TOLERANCE,
        "metrics": {},
        "limits": [
            "This is a 10+10 question synthetic development/holdout benchmark: too small for external generalization claims.",
            "Holdout was not used for tuning. Do not adjust thresholds after seeing holdout results.",
            "Adaptive latency is calculated from measured retrieval/rerank component costs, not direct API timing.",
            "This evaluates retrieval ranking only, not answer correctness/refusal or end-to-end generation latency.",
            "Any benefit from adaptive gating must be confirmed with more diverse independent documents/questions.",
        ],
    }
    rows = []
    for split in ("DEV", "HOLDOUT"):
        summary["metrics"][split] = {}
        for name in ("Hybrid-RRF", "Hybrid+Rerank"):
            summary["metrics"][split][name] = baseline_summary(records, split, name)
        adaptive_sum, adaptive_items = evaluate_policy(records, chosen_gate, split)
        summary["metrics"][split]["Adaptive"] = adaptive_sum
        ad_by_id = {r["id"]: r for r in adaptive_items}
        for r in (r for r in records if r["split"] == split):
            ad = ad_by_id[r["id"]]
            rows.append({
                "question_id": r["id"], "split": split, "question": r["question"],
                "gold_page": r["gold_page"], "top1_agreement": int(r["features"]["top1_agreement"]),
                "bm25_relative_margin": round(r["features"]["bm25_relative_margin"], 5),
                "cosine_relative_margin": round(r["features"]["cosine_relative_margin"], 5),
                "skip_rerank": int(ad["skipped"]),
                "hybrid_rrf_gold_rank": r["rrf_rank"],
                "hybrid_rerank_gold_rank": r["rerank_rank"],
                "adaptive_gold_rank": ad["rank"],
                "adaptive_hit_at_3": ad["hit_at_3"],
                "adaptive_mrr_at_8": round(ad["mrr_at_8"], 6),
                "retrieval_ms": round(r["retrieval_ms"], 3),
                "rerank_ms": round(r["rerank_ms"], 3),
                "estimated_adaptive_ms": round(ad["estimated_ms"], 3),
            })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "adaptive_comparison.csv"
    json_path = OUTPUT / "adaptive_summary.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    print("\n================ ADAPTIVE RERANK EXPERIMENT ================")
    for split in ("DEV", "HOLDOUT"):
        print(f"{split} (n=10; threshold {'selected' if split=='DEV' else 'frozen'}):")
        print(f"{'Method':<19}{'Hit@3':>9}{'MRR@8':>9}{'Rerank%':>10}{'Est.ms':>10}")
        for name in ("Hybrid-RRF", "Hybrid+Rerank", "Adaptive"):
            m = summary["metrics"][split][name]
            print(f"{name:<19}{m['hit_at_3']:>9.1%}{m['mrr_at_8']:>9.3f}"
                  f"{m['rerank_rate']:>10.0%}{m['estimated_latency_ms']:>10.2f}")
    print(f"\nSaved: {csv_path}\nSaved: {json_path}")
    print("NOTE: Small synthetic benchmark; estimated latency is NOT measured API latency.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    hybrid_self_test()
    gate_self_test()
    chunks = load_chunks()
    gold = validate_labels(chunks)
    print(f"Validated {len(TESTS)} questions against {len(chunks)} PDF chunks.")
    print("DEV: 10 odd IDs; HOLDOUT: 10 even IDs; labels only used to score and tune DEV.")
    if not args.validate_only:
        run(chunks, gold)


if __name__ == "__main__":
    main()
