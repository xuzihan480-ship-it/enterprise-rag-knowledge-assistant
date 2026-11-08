"""Calibrate and evaluate an evidence-sufficiency/refusal heuristic.

This test is small and synthetic. Positive questions come from the same
handbook benchmark used in previous experiments, so it is not independent
external validation. Negative questions were authored to have no specified
answer in the demonstration handbook. DEV alone chooses thresholds.

Run:
    python evidence_guard.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false python evaluate_refusal.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from evaluate import TESTS, load_chunks, validate_labels
from hybrid_retrieval import BM25Index, rrf_fuse
from evidence_guard import (
    EvidenceThresholds, evidence_features, has_enough_evidence, self_test,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "eval_results"

# Questions deliberately not answered in the 10-page fictional handbook.
# This is an authored test set, not an automatically verified gold standard.
NEGATIVES = [
    ("N01", "Who is the current chief executive officer of NDS?"),
    ("N02", "What is the legal company registration number for NDS?"),
    ("N03", "What is the headquarters' complete postal address?"),
    ("N04", "What is the company's current share price?"),
    ("N05", "Which university did the AI department leader graduate from?"),
    ("N06", "What is the official customer support telephone number?"),
    ("N07", "What is the exact monthly rent for the headquarters office?"),
    ("N08", "What is the employee Wi-Fi password?"),
    ("N09", "Who is the head of the human resources department?"),
    ("N10", "What was the company's annual revenue in 2025?"),
]

# Explicit grid, no thresholds from held-out labels.
COSINE_GRID = [round(x / 100, 2) for x in range(15, 71, 5)]
COVERAGE_GRID = [round(x / 100, 2) for x in range(0, 101, 10)]


def split_of(qid: str) -> str:
    return "DEV" if int(qid[1:]) % 2 else "HOLDOUT"


def summarize(rows: list[dict], gate: EvidenceThresholds, split: str) -> dict:
    portion = [r for r in rows if r["split"] == split]
    positives = [r for r in portion if r["answerable"]]
    negatives = [r for r in portion if not r["answerable"]]
    pos_accept = mean(has_enough_evidence(r["features"], gate) for r in positives)
    neg_reject = mean(not has_enough_evidence(r["features"], gate) for r in negatives)
    return {
        "positive_acceptance_rate": round(pos_accept, 4),
        "negative_rejection_rate": round(neg_reject, 4),
        "balanced_accuracy": round((pos_accept + neg_reject) / 2, 4),
        "positive_questions": len(positives),
        "negative_questions": len(negatives),
    }


def choose_gate(rows: list[dict]) -> EvidenceThresholds:
    scored = []
    for cos in COSINE_GRID:
        for cov in COVERAGE_GRID:
            gate = EvidenceThresholds(cos, cov)
            s = summarize(rows, gate, "DEV")
            # Prefer not to refuse known-answer DEV questions; among ties,
            # choose stronger negative rejection, then a less strict gate.
            feasible = s["positive_acceptance_rate"] >= 0.8
            scored.append((int(feasible), s["balanced_accuracy"],
                           s["positive_acceptance_rate"],
                           s["negative_rejection_rate"], -cos, -cov, gate))
    return max(scored, key=lambda row: row[:6])[-1]


def main():
    import faiss
    from sentence_transformers import SentenceTransformer

    self_test()
    chunks = load_chunks()
    gold = validate_labels(chunks)
    print(f"Gold checks passed: {len(gold)} positives, {len(chunks)} PDF chunks.", flush=True)
    texts = [c["text"] for c in chunks]
    bm25 = BM25Index(texts)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    vectors = embedder.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    cases = [{"id": t["id"], "question": t["question"], "answerable": True}
             for t in TESTS]
    cases += [{"id": qid, "question": q, "answerable": False} for qid, q in NEGATIVES]
    rows = []
    for case in cases:
        q = case["question"]
        qvec = embedder.encode([q], normalize_embeddings=True).astype("float32")
        sims, ids = index.search(qvec, min(8, len(chunks)))
        dense_ids = [int(i) for i in ids[0]]
        bm_scores = bm25.score(q)
        bm_ids = sorted(range(len(chunks)), key=lambda i: (-bm_scores[i], i))[:8]
        rrf_ids = rrf_fuse(dense_ids, bm_ids, top_k=8)
        features = evidence_features(q, float(sims[0][0]),
                                     [texts[i] for i in rrf_ids[:3]])
        rows.append({"id": case["id"], "split": split_of(case["id"]),
                     "question": q, "answerable": case["answerable"],
                     "features": features})

    gate = choose_gate(rows)
    dev = summarize(rows, gate, "DEV")
    holdout = summarize(rows, gate, "HOLDOUT")
    print("\nChosen using DEV only:", gate)
    for split, m in (("DEV", dev), ("HOLDOUT", holdout)):
        print(f"{split} (positive n={m['positive_questions']}, negative n={m['negative_questions']}): ")
        print(f"  Known-answer acceptance: {100*m['positive_acceptance_rate']:.1f}%")
        print(f"  Unknown-answer rejection: {100*m['negative_rejection_rate']:.1f}%")
        print(f"  Balanced accuracy:        {100*m['balanced_accuracy']:.1f}%")

    for row in rows:
        flag = has_enough_evidence(row["features"], gate)
        row["decision"] = "answer" if flag else "abstain"
        row["correct"] = flag == row["answerable"]

    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "evidence_gate_config.json"
    path.write_text(json.dumps({"min_cosine": gate.min_cosine,
                                "min_coverage": gate.min_coverage,
                                "selected_on": "DEV (10 positive + 5 negative questions)",
                                "notes": "Heuristic answerability filter; not verified factual support"},
                               indent=2), encoding="utf-8")
    (OUTPUT / "refusal_evaluation.json").write_text(
        json.dumps({"design": "Small synthetic; prior positive questions previously inspected",
                    "dev": dev, "holdout": holdout, "cases": rows},
                   indent=2), encoding="utf-8")
    print("\nSaved:", path)
    print("Saved:", OUTPUT / "refusal_evaluation.json")
    print("\nLIMITATION: Similarity and keyword matches cannot prove factual support.")


if __name__ == "__main__":
    main()
