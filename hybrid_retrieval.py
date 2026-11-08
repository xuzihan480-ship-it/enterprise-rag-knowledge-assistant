"""BM25 and Reciprocal Rank Fusion for the Enterprise Knowledge Assistant.

Pure-Python implementation: no extra pip package is needed.
This module does NOT change the existing FastAPI app.
"""

from collections import Counter
import math
import re


def tokenize(text: str) -> list[str]:
    """Simple English tokenization for the demonstration handbook."""
    return re.findall(r"[a-z0-9]+", text.casefold())


class BM25Index:
    """Okapi BM25 lexical retriever with chunk IDs matching the FAISS index."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        if not texts:
            raise ValueError("BM25 requires at least one document chunk")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(text)) for text in texts]
        self.doc_lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.count = len(texts)

        doc_freq = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        self.idf = {
            term: math.log(1 + (self.count - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score(self, query: str) -> list[float]:
        scores = [0.0] * self.count
        # Duplicate words in a question should not accidentally dominate ranking.
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in enumerate(self.term_freqs):
                frequency = tf.get(term, 0)
                if frequency == 0:
                    continue
                length_norm = 1 - self.b + self.b * self.doc_lengths[doc_id] / self.avg_length
                scores[doc_id] += idf * (
                    frequency * (self.k1 + 1)
                    / (frequency + self.k1 * length_norm)
                )
        return scores

    def search(self, query: str, top_k: int = 8) -> list[int]:
        scores = self.score(query)
        return sorted(range(self.count), key=lambda i: (-scores[i], i))[:top_k]


def rrf_fuse(
    semantic_ids: list[int], lexical_ids: list[int],
    top_k: int = 8, rrf_k: int = 60,
) -> list[int]:
    """Merge two ranked lists. Larger RRF scores imply higher combined rank.

    Both source lists should be sorted best-first and contain unique chunk IDs.
    RRF uses ranks, not incompatible BM25 and cosine raw scores.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    fused = {}
    for ids in (semantic_ids, lexical_ids):
        for rank, chunk_id in enumerate(ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (rrf_k + rank)
    # Deterministic tie-breaking by chunk ID.
    return sorted(fused, key=lambda i: (-fused[i], i))[:top_k]


def self_test() -> None:
    docs = [
        "Security policy defines access permissions.",
        "Data quality uses schema validation and consistency checks.",
        "Inference latency and GPU throughput are monitored.",
    ]
    idx = BM25Index(docs)
    assert idx.search("GPU latency", 1) == [2]
    assert idx.search("schema consistency", 1) == [1]
    fused = rrf_fuse([0, 1, 2], [2, 1, 0], top_k=3)
    assert len(fused) == 3 and len(set(fused)) == 3
    assert rrf_fuse([0, 1], [0, 2], 3)[0] == 0
    print("BM25 + RRF self-test passed.")


if __name__ == "__main__":
    self_test()
