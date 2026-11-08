"""Conservative, transparent evidence-sufficiency heuristic for local RAG.

This is an *answerability screening* rule, not a proof that generated
claims are true. A high cosine similarity alone is not enough: the query
must also have lexical support among retrieved chunks. Calibrate thresholds
on DEV only; test HOLDOUT separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from hybrid_retrieval import tokenize

# Generic question words: removing them stops e.g. "what is the" from
# spuriously inflating lexical coverage.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
    "could", "did", "do", "does", "each", "for", "from", "had", "has", "have",
    "how", "i", "if", "in", "into", "is", "it", "its", "many", "much", "must",
    "of", "on", "or", "our", "should", "so", "than", "that", "the", "their",
    "them", "there", "these", "this", "to", "under", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "please", "tell", "me", "about", "give", "name", "list", "some",
})


@dataclass(frozen=True)
class EvidenceThresholds:
    min_cosine: float
    min_coverage: float


def informative_terms(question: str) -> set[str]:
    return {t for t in tokenize(question) if t not in STOPWORDS and len(t) > 1}


def evidence_features(question: str, best_cosine: float, evidence_texts: Iterable[str]) -> dict:
    terms = informative_terms(question)
    evidence_terms = set()
    for text in evidence_texts:
        evidence_terms.update(tokenize(text))
    covered = terms & evidence_terms
    return {
        "best_cosine": round(float(best_cosine), 6),
        "query_term_coverage": round(len(covered) / len(terms), 6) if terms else 0.0,
        "query_term_count": len(terms),
        "matched_term_count": len(covered),
    }


def has_enough_evidence(features: dict, threshold: EvidenceThresholds) -> bool:
    return (
        features["query_term_count"] > 0
        and features["best_cosine"] >= threshold.min_cosine
        and features["query_term_coverage"] >= threshold.min_coverage
    )


def self_test() -> None:
    cfg = EvidenceThresholds(min_cosine=0.3, min_coverage=0.5)
    yes = evidence_features(
        "What are the five data quality dimensions?", 0.67,
        ["The five data quality dimensions are completeness and uniqueness."]
    )
    assert has_enough_evidence(yes, cfg)
    no = evidence_features("Who is the company CEO?", 0.21,
                           ["Data quality involves completeness."])
    assert not has_enough_evidence(no, cfg)
    assert not has_enough_evidence(evidence_features("the and an", 0.99, ["x"]), cfg)
    print("Evidence guard self-test passed.")


if __name__ == "__main__":
    self_test()
