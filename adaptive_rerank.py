"""Confidence-gated reranking on a hybrid (BM25 + FAISS -> RRF) candidate list.

Only pre-rerank features are allowed in the decision rule. This module never
uses ground-truth labels at inference time.
"""
from dataclasses import dataclass


def relative_margin(first: float, second: float, scale_floor: float = 1.0) -> float:
    """Nonnegative margin normalized by the magnitude of the leading score."""
    return max(0.0, first - second) / max(scale_floor, abs(first))


@dataclass(frozen=True)
class Gate:
    bm25_margin_min: float = 0.05
    cosine_margin_min: float = 0.02


def gate_features(
    faiss_ids: list[int],
    faiss_scores: list[float],
    bm25_ids: list[int],
    bm25_scores: list[float],
) -> dict:
    """Create features using raw retrieval outputs, before any cross-encoder call.

    All score and ID lists must use best-first order. Compare the two retrieval
    systems' rank-1 chunks; their raw scores are *not* compared to each other.
    """
    if min(len(faiss_ids), len(faiss_scores), len(bm25_ids), len(bm25_scores)) < 2:
        raise ValueError("At least two results required from each retriever")
    return {
        "top1_agreement": bool(faiss_ids[0] == bm25_ids[0]),
        "bm25_relative_margin": relative_margin(bm25_scores[0], bm25_scores[1]),
        "cosine_relative_margin": relative_margin(faiss_scores[0], faiss_scores[1]),
    }


def skip_rerank(features: dict, gate: Gate) -> bool:
    """Skip the expensive reranker only when *both* retrievers look confident.

    Conservative heuristic, not a probability estimate or calibrated confidence.
    """
    return (
        features["top1_agreement"]
        and features["bm25_relative_margin"] >= gate.bm25_margin_min
        and features["cosine_relative_margin"] >= gate.cosine_margin_min
    )


def self_test():
    features = gate_features(
        faiss_ids=[2, 1], faiss_scores=[0.8, 0.5],
        bm25_ids=[2, 0], bm25_scores=[8.0, 3.0]
    )
    assert features["top1_agreement"]
    assert skip_rerank(features, Gate())
    assert not skip_rerank(features, Gate(cosine_margin_min=0.5))
    assert not skip_rerank({**features, "top1_agreement": False}, Gate())
    assert relative_margin(5, 5) == 0
    print("Adaptive rerank gate self-test passed.")


if __name__ == "__main__":
    self_test()
